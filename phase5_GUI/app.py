"""
Invigilo — AI-Assisted Exam Proctoring System
Flask web application wrapping the Phase 4 v3 inference pipeline.
"""

import os, sys, json, time, uuid, threading, csv as csv_mod, io
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from flask import (
    Flask, render_template, request, jsonify, Response,
    send_from_directory, redirect, url_for,
)

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent))

try:
    from phase4_inference_v3 import (
        InferenceEngine, FLAG_THRESHOLD, FLAG_WINDOW,
        BASELINE_ENABLED, HAS_SUPERVISION, MIN_TRACK_OBSERVATIONS,
        BYTETRACK_LOST_BUFFER, PersonTracker,
    )
    INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f'WARNING: Could not import phase4_inference_v3: {e}')
    INFERENCE_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(APP_DIR / 'uploads')
app.config['OUTPUT_FOLDER'] = str(APP_DIR / 'output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

engine = None
engine_lock = threading.Lock()
jobs = {}
jobs_lock = threading.Lock()
DEFAULT_PKG = str(APP_DIR / 'inference_package_v11.pth')


def get_engine(pkg_path=None):
    global engine
    with engine_lock:
        if engine is None:
            pkg = pkg_path or DEFAULT_PKG
            if not os.path.exists(pkg):
                raise FileNotFoundError(f'Inference package not found: {pkg}')
            engine = InferenceEngine(pkg)
        return engine


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


# ---------------------------------------------------------------------------
# Inference thread — processes frames, stores latest for MJPEG + events
# ---------------------------------------------------------------------------

def run_monitor_job(job_id):
    """Process video/webcam frame-by-frame, storing frames and events for live streaming."""
    job = jobs[job_id]
    try:
        job['status'] = 'loading_engine'
        eng = get_engine()

        # Wait for start signal
        job['status'] = 'ready'
        while job.get('control') != 'start':
            if job.get('control') == 'abort':
                job['status'] = 'aborted'
                return
            time.sleep(0.1)

        job['status'] = 'running'
        job['start_time'] = time.time()
        source = job.get('video_path') or 0  # 0 = webcam

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video source: {source}')

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        job['video_fps'] = fps
        job['video_total_frames'] = total_frames
        job['video_width'] = w
        job['video_height'] = h

        # Reset engine
        eng.tracker = PersonTracker(fps=fps)
        eng.person_states = {}
        eng.merge_count = 0

        # Output paths
        output_dir = Path(job['output_dir'])
        stem = job['video_stem']
        out_video_path = output_dir / f'{stem}_annotated.mp4'
        timeline_path = output_dir / f'{stem}_timeline.csv'
        summary_path = output_dir / f'{stem}_summary.txt'

        # Video writer — try H.264 first for browser compat
        writer = None
        for codec in ['avc1', 'H264', 'mp4v', 'XVID']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
            if writer.isOpened():
                job['codec'] = codec
                break
            writer = None
        if writer is None:
            raise RuntimeError('No working video codec found')

        # Timeline CSV
        tl_file = open(timeline_path, 'w', newline='', encoding='utf-8')
        tl_writer = csv_mod.writer(tl_file)
        tl_writer.writerow(['time_sec', 'frame', 'track_id', 'label', 'prob',
                            'is_flagged', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])

        frame_idx = 0
        last_log_sec = -1
        proc_start = time.time()
        prev_flagged_ids = set()

        while job.get('control') != 'stop':
            ret, frame = cap.read()
            if not ret:
                break

            results = eng.process_frame(frame, fps)
            annotated = eng.draw_annotations(frame, results, frame_idx, fps)
            writer.write(annotated)

            # Encode for MJPEG stream
            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            job['latest_frame'] = jpeg.tobytes()

            # Track current state
            cur_sec = frame_idx / fps
            people_count = len(results)
            sus_results = [r for r in results if r['label'] != 'normal'
                           and not r.get('is_new') and not r.get('is_calibrating')
                           and not r.get('is_warming', False)]
            flagged_results = [r for r in results if r.get('is_flagged')]

            # Update live stats
            job['live_stats'] = {
                'elapsed': fmt_time(cur_sec),
                'elapsed_sec': round(cur_sec, 1),
                'frame': frame_idx,
                'total_frames': total_frames,
                'progress': round(frame_idx / max(total_frames, 1) * 100, 1) if total_frames > 0 else 0,
                'people': people_count,
                'suspicious': len(sus_results),
                'flagged': len(flagged_results),
                'fps': round(frame_idx / max(time.time() - proc_start, 0.1), 1),
            }

            # Track active flags for the panel
            cur_flagged_ids = set()
            active_flags = []
            for r in flagged_results:
                tid = r['track_id']
                cur_flagged_ids.add(tid)
                active_flags.append({
                    'person': f'P{tid + 1}',
                    'track_id': tid,
                    'label': r['label'],
                    'prob': round(r['prob'], 2),
                    'time': fmt_time(cur_sec),
                })
            job['active_flags'] = active_flags

            # Log events (1 per second per person)
            log_sec = int(cur_sec)
            if log_sec > last_log_sec:
                last_log_sec = log_sec
                for r in results:
                    if r.get('is_new') or r.get('is_calibrating') or r.get('is_warming', False):
                        continue
                    tid = r['track_id']
                    bbox = r['bbox']
                    tl_writer.writerow([
                        round(cur_sec, 1), frame_idx, tid,
                        r['label'], round(r['prob'], 3), r['is_flagged'],
                        bbox[0], bbox[1], bbox[2], bbox[3],
                    ])
                    # Add to event log if suspicious
                    if r['label'] != 'normal':
                        job['event_log'].append({
                            'time': fmt_time(cur_sec),
                            'time_sec': round(cur_sec, 1),
                            'person': f'P{tid + 1}',
                            'label': r['label'],
                            'prob': round(r['prob'], 2),
                            'flagged': r.get('is_flagged', False),
                        })

                # New flag events
                new_flags = cur_flagged_ids - prev_flagged_ids
                for tid in new_flags:
                    r = next((x for x in flagged_results if x['track_id'] == tid), None)
                    if r:
                        job['flag_events'].append({
                            'time': fmt_time(cur_sec),
                            'person': f'P{tid + 1}',
                            'label': r['label'],
                            'prob': round(r['prob'], 2),
                            'type': 'flag_start',
                        })
                prev_flagged_ids = cur_flagged_ids

            frame_idx += 1
            job['frames_processed'] = frame_idx

        # Cleanup
        cap.release()
        writer.release()
        tl_file.close()

        elapsed = time.time() - proc_start
        fps_proc = frame_idx / max(elapsed, 0.1)

        # Write summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f'Invigilo Inference Summary\n{"=" * 30}\n\n')
            f.write(f'Video: {job.get("video_name", "webcam")}\n')
            f.write(f'Duration: {frame_idx / fps:.1f} seconds\n')
            f.write(f'Frames: {frame_idx}\n')
            f.write(f'Speed: {fps_proc:.1f} fps\n')
            f.write(f'Time: {elapsed:.1f} seconds\n')
            f.write(f'Threshold: {eng.threshold:.2f}\n')
            f.write(f'Flagging rule: {FLAG_THRESHOLD}-of-{FLAG_WINDOW}\n')
            f.write(f'Baseline calibration: {"ON" if BASELINE_ENABLED else "OFF"}\n')
            f.write(f'Tracker: {"ByteTrack" if HAS_SUPERVISION else "IoU fallback"}\n')
            f.write(f'Track merges: {eng.merge_count}\n')

        # Store output file info
        job['outputs'] = {
            'annotated_video': str(out_video_path),
            'timeline_csv': str(timeline_path),
            'summary_txt': str(summary_path),
            'annotated_video_name': f'{stem}_annotated.mp4',
            'timeline_csv_name': f'{stem}_timeline.csv',
            'summary_txt_name': f'{stem}_summary.txt',
        }
        job['end_time'] = time.time()
        job['status'] = 'complete'
        job['processing_time'] = round(elapsed, 1)
        job['video_duration'] = round(frame_idx / fps, 1)

    except Exception as e:
        import traceback
        job['status'] = 'error'
        job['error'] = str(e)
        job['traceback'] = traceback.format_exc()
        print(f'ERROR in job {job_id}: {e}')
        traceback.print_exc()


def build_results_data(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'complete':
        return None

    data = {
        'job_id': job_id,
        'video_name': job.get('video_name', 'webcam'),
        'processing_time': job.get('processing_time', 0),
        'video_duration': job.get('video_duration', 0),
    }

    # Parse timeline for stats
    tl_path = job['outputs'].get('timeline_csv')
    if tl_path and os.path.exists(tl_path):
        rows = []
        with open(tl_path, 'r', encoding='utf-8') as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                rows.append(row)

        persons = {}
        for r in rows:
            pid = int(r['track_id'])
            if pid not in persons:
                persons[pid] = {
                    'track_id': pid, 'label': f'P{pid + 1}',
                    'total': 0, 'suspicious': 0, 'flagged': 0, 'classes': {},
                }
            persons[pid]['total'] += 1
            if r['label'] != 'normal':
                persons[pid]['suspicious'] += 1
                cls = r['label']
                persons[pid]['classes'][cls] = persons[pid]['classes'].get(cls, 0) + 1
            if r['is_flagged'] in ('True', 'true', '1', 'yes'):
                persons[pid]['flagged'] += 1

        data['persons'] = sorted(persons.values(), key=lambda p: p['track_id'])
        total_obs = len(rows)
        sus_obs = sum(1 for r in rows if r['label'] != 'normal')
        flag_obs = sum(1 for r in rows if r['is_flagged'] in ('True', 'true', '1', 'yes'))
        data['stats'] = {
            'total_observations': total_obs,
            'suspicious_observations': sus_obs,
            'flagged_observations': flag_obs,
            'suspicious_rate': round(sus_obs / max(total_obs, 1) * 100, 1),
            'flagged_rate': round(flag_obs / max(total_obs, 1) * 100, 1),
        }

    # Summary text
    summary_path = job['outputs'].get('summary_txt')
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            data['summary'] = f.read()

    return data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    status = {
        'inference_available': INFERENCE_AVAILABLE,
        'engine_loaded': engine is not None,
        'pkg_exists': os.path.exists(DEFAULT_PKG),
    }
    return render_template('index.html', status=status)


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if not INFERENCE_AVAILABLE:
        return jsonify({'error': 'Inference engine not available'}), 500
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    safe_name = f'{uuid.uuid4().hex[:8]}_{file.filename}'
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    file.save(upload_path)

    job_id = uuid.uuid4().hex[:12]
    job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    stem = Path(safe_name).stem

    with jobs_lock:
        jobs[job_id] = {
            'id': job_id, 'status': 'initializing', 'control': None,
            'mode': 'video', 'video_name': file.filename, 'video_stem': stem,
            'video_path': upload_path, 'output_dir': job_output_dir,
            'latest_frame': None, 'live_stats': {},
            'active_flags': [], 'event_log': [], 'flag_events': [],
            'frames_processed': 0, 'outputs': {},
            'start_time': None, 'end_time': None, 'error': None,
        }

    t = threading.Thread(target=run_monitor_job, args=(job_id,), daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'redirect': url_for('monitor', job_id=job_id)})


@app.route('/webcam')
def start_webcam():
    if not INFERENCE_AVAILABLE:
        return render_template('error.html', message='Inference engine not available'), 500

    job_id = uuid.uuid4().hex[:12]
    job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with jobs_lock:
        jobs[job_id] = {
            'id': job_id, 'status': 'initializing', 'control': None,
            'mode': 'webcam', 'video_name': f'webcam_{stamp}',
            'video_stem': f'webcam_{stamp}', 'video_path': None,
            'output_dir': job_output_dir,
            'latest_frame': None, 'live_stats': {},
            'active_flags': [], 'event_log': [], 'flag_events': [],
            'frames_processed': 0, 'outputs': {},
            'start_time': None, 'end_time': None, 'error': None,
        }

    t = threading.Thread(target=run_monitor_job, args=(job_id,), daemon=True)
    t.start()

    return redirect(url_for('monitor', job_id=job_id))


@app.route('/monitor/<job_id>')
def monitor(job_id):
    job = jobs.get(job_id)
    if not job:
        return render_template('error.html', message='Session not found'), 404
    return render_template('monitor.html', job=job)


@app.route('/api/start/<job_id>', methods=['POST'])
def api_start(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    job['control'] = 'start'
    return jsonify({'ok': True})


@app.route('/api/stop/<job_id>', methods=['POST'])
def api_stop(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    job['control'] = 'stop'
    return jsonify({'ok': True})


@app.route('/stream/<job_id>')
def video_stream(job_id):
    """MJPEG stream of annotated frames."""
    def generate():
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'Waiting for Start...', (140, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        _, ph_jpg = cv2.imencode('.jpg', placeholder)
        ph_bytes = ph_jpg.tobytes()

        while True:
            job = jobs.get(job_id)
            if not job:
                break
            frame_data = job.get('latest_frame')
            if frame_data:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            else:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + ph_bytes + b'\r\n')

            if job.get('status') in ('complete', 'error', 'aborted'):
                # Send final frame a few more times then stop
                for _ in range(3):
                    if frame_data:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                    time.sleep(0.3)
                break
            time.sleep(0.06)  # ~16fps display rate

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/state/<job_id>')
def api_state(job_id):
    """Poll endpoint for live stats, flags, and events."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404

    last_event_idx = request.args.get('last_event', 0, type=int)
    new_events = job.get('event_log', [])[last_event_idx:]

    return jsonify({
        'status': job['status'],
        'stats': job.get('live_stats', {}),
        'flags': job.get('active_flags', []),
        'new_events': new_events[-20:],  # Cap per poll
        'total_events': len(job.get('event_log', [])),
        'error': job.get('error'),
    })


@app.route('/results/<job_id>')
def results(job_id):
    job = jobs.get(job_id)
    if not job:
        return render_template('error.html', message='Session not found'), 404
    if job['status'] != 'complete':
        return redirect(url_for('monitor', job_id=job_id))

    data = build_results_data(job_id)
    return render_template('results.html', data=data, job=job)


@app.route('/output/<job_id>/<filename>')
def serve_output(job_id, filename):
    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    fpath = os.path.join(job_dir, filename)
    if not os.path.exists(fpath):
        return 'File not found', 404
    return send_from_directory(job_dir, filename)


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum 500 MB.'}), 413


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message=str(e)), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Invigilo — AI-Assisted Exam Proctoring')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--pkg', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.pkg:
        DEFAULT_PKG = args.pkg

    print(f'\n{"=" * 50}')
    print(f'  INVIGILO — AI-Assisted Exam Proctoring')
    print(f'{"=" * 50}')
    print(f'  Engine: {"Ready" if INFERENCE_AVAILABLE else "NOT AVAILABLE"}')
    print(f'  Package: {DEFAULT_PKG} ({"found" if os.path.exists(DEFAULT_PKG) else "NOT FOUND"})')
    print(f'\n  Open http://{args.host}:{args.port}\n')

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
