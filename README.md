# Invigilo

**AI-Assisted Suspicious Behaviour Detection in Physical Exam Rooms Using Computer Vision**

Invigilo is a real-time exam proctoring system that uses pose-based computer vision to detect suspicious behaviour in physical examination halls. It processes a room-level camera feed, analyses every student's body posture, and alerts the invigilator when someone's behaviour is consistently suspicious — giving human proctors a second pair of eyes that never blinks.

Built as a group project for **42028: Deep Learning and CNN** at the University of Technology Sydney (Project 46, Team Invigilo).

---

## How It Works

A fixed camera watches the exam room. Each frame passes through a five-stage pipeline:

1. **Person Detection** — YOLOv8-nano locates every student in the frame
2. **Tracking** — ByteTrack assigns stable IDs across frames with a graveyard-merge heuristic for track recovery
3. **Pose Estimation** — MediaPipe Pose extracts 33 body keypoints per person
4. **Feature Engineering** — 13 geometric features (head turn angle, torso lean, wrist position, ear asymmetry, etc.) are computed and aggregated over 3-second sliding windows into 78-dimensional vectors
5. **Classification & Flagging** — A two-stage MLP classifies normal vs suspicious, then head_turn vs lateral_movement. The 4-of-7 rule flags only sustained suspicious behaviour

The output is a live annotated video feed with colour-coded bounding boxes: green for normal, red for flagged.

## Results

Evaluated on 9 unseen test videos (107 suspicious events, 36 hard negatives):

| Metric | Value |
|---|---|
| Event detection rate | 105/107 (98.1%) |
| Flagging rate (4-of-7) | 73/107 (68.2%) |
| Missed events | 2/107 (1.9%) |
| Class accuracy | 57/105 (54.3%) |

The system reliably detects that suspicious behaviour is occurring (98.1%) but has limited ability to distinguish between head_turn and lateral_movement due to overlapping pose features between the two classes.

## Repository Structure

```
Invigilo/
├── data/
│   └── 05_annotations_csv/          # Ground truth annotations and clip labels
├── docs/                             # Project documentation and reports
├── model/                            # Trained model weights (.pth)
├── phase1_dataExtraction_processing_scripts/
│   └── ...                           # YOLOv8 + MediaPipe pose extraction from raw clips
├── phase2_feature_extraction_notebooks/
│   └── ...                           # Feature engineering: keypoints → 78-dim vectors
├── phase3_training_notebooks/
│   └── ...                           # Model training (v8–v14 experiments)
├── phase4_inference/
│   ├── phase4_inference_v3.py        # Final inference pipeline (ByteTrack + graveyard merge)
│   └── evaluate_inference.py         # Evaluation: timeline CSV vs ground truth
└── phase5_GUI/
    ├── app.py                        # Flask web application
    ├── templates/                    # HTML templates (landing, upload, monitor, results)
    └── static/                       # CSS, JS, assets
```

## Setup

### Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- MediaPipe 0.10.14 (requires `numpy<2`)

### Installation

```bash
git clone https://github.com/vaibhavbai1/Invigilo.git
cd Invigilo

pip install torch torchvision
pip install mediapipe==0.10.14
pip install ultralytics opencv-python flask
pip install supervision scikit-learn pandas matplotlib
```

> **Note:** MediaPipe 0.10.14 is incompatible with NumPy 2.x. If you encounter errors, run `pip install "numpy<2"`.

### Model Weights

Place the inference package in the appropriate directory:

```
model/inference_package_v11.pth
```

This file contains the trained Stage 1 (normal vs suspicious) and Stage 2 (head_turn vs lateral_movement) MLP weights, feature scaling parameters, and class mappings.

## Usage

### Web Application (Phase 5)

```bash
cd phase5_GUI
python app.py
```

Open `http://localhost:5000` in your browser. You can either upload a recorded exam video or connect a live webcam for real-time monitoring.

The monitor dashboard shows:
- Live annotated video with colour-coded bounding boxes
- Flagged events panel with person ID, behaviour class, and confidence
- Scrollable event log of all suspicious detections
- Download buttons for annotated video, timeline CSV, and summary report

### Command-Line Inference (Phase 4)

```bash
# Single video
python phase4_inference/phase4_inference_v3.py \
    --video test.mp4 \
    --pkg model/inference_package_v11.pth

# Directory of videos
python phase4_inference/phase4_inference_v3.py \
    --video_dir test_videos/ \
    --pkg model/inference_package_v11.pth --show

# Live webcam
python phase4_inference/phase4_inference_v3.py \
    --webcam \
    --pkg model/inference_package_v11.pth
```

Outputs per video:
- `{video}_annotated.mp4` — video with bounding boxes
- `{video}_timeline.csv` — per-person, per-second classification data
- `{video}_summary.txt` — processing metadata

### Evaluation

```bash
# Single video evaluation
python phase4_inference/evaluate_inference.py \
    --timeline TV01_timeline.csv \
    --ground_truth data/05_annotations_csv/long_video_event_log.xlsx \
    --video_id TV01

# All videos
python phase4_inference/evaluate_inference.py \
    --timeline_dir ./output/ \
    --ground_truth data/05_annotations_csv/long_video_event_log.xlsx
```

## Pipeline Details

### Feature Engineering (13 features)

| Feature | What it captures |
|---|---|
| Head turn angle | Lateral nose displacement from shoulder midpoint |
| Head turn velocity | Speed of head rotation (scanning vs steady gaze) |
| Nose-to-shoulder distance | How far the head has dropped |
| Torso lean angle | Upper body angle from vertical |
| Shoulder tilt | Asymmetric shoulder elevation |
| Left/right wrist drop | Hand positions relative to shoulders |
| Left/right wrist extent | Lateral arm reach |
| Ear visibility asymmetry | Far ear occlusion when head is turned |
| Mouth movement | Frame-to-frame lip displacement |
| Body movement | Average displacement of 5 key joints |

Each feature is normalised by shoulder width (camera-distance invariant), buffered in 3-second sliding windows, and summarised into 6 statistics (mean, std, max, min, range, zero-crossings) → **78-dimensional vector**.

### Model Architecture (v11)

Two-stage MLP, both stages identical: `78 → 128 → 64 → 2` with BatchNorm, ReLU, 40% dropout.

- **Stage 1:** Normal vs suspicious (threshold 0.20, tuned for high recall)
- **Stage 2:** Head turn vs lateral movement (runs only when Stage 1 predicts suspicious)
- **Flagging:** 4-of-7 rule — flagged only if suspicious in ≥4 of last 7 windows

### Training Experiments

| Version | Architecture | Classes | Test Acc | Macro F1 |
|---|---|---|---|---|
| v8 | Two-stage MLP | 5 suspicious | 75.7% | 0.474 |
| v9 | Two-stage MLP | 3 merged | 78.8% | 0.634 |
| **v11** | **Two-stage MLP** | **2 sus (deployed)** | **82.4%** | **0.691** |
| v12 | 1D-CNN temporal | 4 classes | 79.8% | 0.683 |
| v13 | 1D-CNN temporal | 3 classes | 83.1% | 0.690 |
| v14 | Bi-GRU temporal | 4 classes | 79.3% | 0.722 |

## Dataset

Custom dataset of **339 annotated 10-second video clips** recorded by our team in a university lecture theatre. 8 students seated in two rows (A1–A4 front, B1–B4 back), one fixed front-facing camera.

**Behaviour classes (original → merged):**
- Looking sideways + talking to neighbour → `head_turn`
- Leaning to neighbour + passing note → `lateral_movement`
- Looking down → dropped in v11 (caused excessive false alarms)
- Using phone → excluded (indistinguishable from writing in pose features)
- Normal + hard negatives (scratching, stretching, page turning) → `normal`

**Test videos:** 9 videos (TV01–TV09), 53–108 seconds each, 5–7 people per video, with annotated suspicious events and hard negatives.

## Known Limitations

- **Class confusion:** The model predicts lateral_movement for ~73% of head_turn events. It works well as a binary normal-vs-suspicious detector but cannot reliably distinguish the two suspicious types.
- **Hard-negative false alarms (~69%):** Normal fidgeting shares the same physical movements as cheating. Pose features alone cannot distinguish intent.
- **Small, homogeneous dataset:** 339 clips of 8 people in one room. Generalisation to different rooms, cameras, or body types is unverified.
- **No object detection:** Phone and note detection relies on posture changes rather than direct visual recognition.
- **CPU inference speed:** 6–8 fps on CPU. GPU acceleration would enable native frame-rate processing.

## Tech Stack

- **Detection:** YOLOv8-nano (Ultralytics)
- **Tracking:** ByteTrack (via Supervision library)
- **Pose:** MediaPipe Pose (33 keypoints)
- **Classifier:** PyTorch (two-stage MLP)
- **Web app:** Flask + MJPEG streaming
- **Frontend:** HTML / CSS / JavaScript
- **Training:** AWS SageMaker (`conda_pytorch_p310`)

## Team

| Name | Student ID |
|---|---|
| Vaibhav Bairathi | 25534645 |
| Praveer Jain | 25947209 |
| Chandan Sreenivasaiah | 25674250 |

## License

This project was developed as a university assignment for educational purposes.
