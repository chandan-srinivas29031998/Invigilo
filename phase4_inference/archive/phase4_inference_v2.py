"""
PHASE 4 v2: Live Inference Pipeline (Improved)
IMPROVEMENTS OVER v1 (based on V01/V02/V03 long-video evaluation):

  FIX 1 — ByteTrack replaces nearest-neighbour tracker
    v1 had 28 track IDs for 8 people (V03), causing 3 missed events
    when feature buffers reset on ID reassignment. ByteTrack uses
    Kalman-predicted IoU matching + 3-second lost-track buffer.
    Falls back to IoU-based matching if supervision is not installed.

  FIX 2 — Per-person baseline calibration
    v1's P2 in V02 was flagged 63% of the time because they naturally
    sit with a slight head tilt. Now the first BASELINE_SEC seconds are
    used to compute each person's resting-posture feature averages.
    Subsequent features are expressed as deviations from that baseline,
    so a naturally tilted head produces near-zero head_turn deviation.

  FIX 3 — Stricter flagging rule (4-of-7 instead of 3-of-5)
    v1's 3-of-5 rule let brief fidgeting (scratching face, turning pages)
    trigger flags. 4-of-7 requires more sustained suspicious behaviour,
    cutting short false-alarm spikes while still catching real cheating.

  FIX 4 — Looking-down suppression
    Sustained looking_down (>3 consecutive windows) is reclassified as
    normal, since it's almost always a student writing. Brief glances
    (1-3 windows) are kept as suspicious.
"""

import os
import sys
import csv
import time
import pickle
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import torch
import mediapipe as mp
from ultralytics import YOLO

# Try to import supervision for ByteTrack; fall back gracefully
try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False
    print('WARNING: supervision not installed — using IoU fallback tracker.')
    print('  Install for better tracking: pip install supervision')


# CONFIGURATION


# --- Model paths ---
DEFAULT_INFERENCE_PKG = './inference_package_v11.pth'

# --- YOLOv8 ---
YOLO_MODEL       = 'yolov8n.pt'
YOLO_CONF        = 0.5
PERSON_CLASS_ID  = 0
MAX_PEOPLE       = 10      

# --- MediaPipe Pose ---
MP_MODEL_COMPLEXITY      = 1
MP_MIN_DETECTION_CONF    = 0.5
MP_MIN_TRACKING_CONF     = 0.5

# --- Feature computation ---
MIN_VISIBILITY   = 0.5
NUM_KEYPOINTS    = 33

# --- Sliding window ---
WINDOW_SIZE_SEC  = 3.0
STRIDE_SEC       = 1.0

# FIX 2: Per-person baseline calibration
BASELINE_ENABLED = True
BASELINE_SEC     = 10.0    # Seconds of calibration at start

# FIX 3: Stricter flagging rule 
FLAG_WINDOW      = 7      
FLAG_THRESHOLD   = 4       

# FIX 4: Looking-down suppression 
LOOKDOWN_SUPPRESS        = True
LOOKDOWN_MAX_CONSECUTIVE = 3   



BYTETRACK_ACTIVATION_THRESHOLD = 0.4
BYTETRACK_LOST_BUFFER          = 90   # ~3 seconds at 30fps
BYTETRACK_MATCHING_THRESHOLD   = 0.8
# IoU fallback parameters (used when supervision is not installed)
IOU_MATCH_THRESHOLD = 0.7    # Minimum IoU to match detection to track
IOU_MAX_MISSING     = 90    

# --- Visualisation ---
COLOR_NORMAL      = (0, 200, 0)      # Green
COLOR_SUSPICIOUS  = (0, 0, 220)      # Red
COLOR_WARNING     = (0, 200, 255)    # Yellow
COLOR_CALIBRATING = (200, 150, 0)    # Blue person is in baseline calibration
COLOR_UNTRACKED   = (180, 180, 180)  # Grey
FONT              = cv2.FONT_HERSHEY_SIMPLEX


# FEATURE COMPUTATION (identical to Phase 2)


def kp_to_frame_coords(kp, bbox):
    """Convert crop-normalised keypoints to frame pixel coordinates."""
    result = kp.copy()
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w > 0 and h > 0:
        result[:, 0] = x1 + kp[:, 0] * w
        result[:, 1] = y1 + kp[:, 1] * h
    return result


def safe_angle(v1, v2):
    """Angle in degrees between two 2D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def compute_frame_features(kp_frame, prev_kp_frame, ref_shoulder_w):
    """Compute 13 pose features from one frame's keypoints in frame coordinates."""
    feat = {}
    vis = kp_frame[:, 3]
    sw = max(ref_shoulder_w, 1.0)

    def pt(idx):
        return kp_frame[idx, :2] if vis[idx] >= MIN_VISIBILITY else None

    nose = pt(0)
    mouth_l, mouth_r = pt(9), pt(10)
    l_sh, r_sh = pt(11), pt(12)
    l_wr, r_wr = pt(15), pt(16)
    l_hip, r_hip = pt(23), pt(24)
    mid_sh = (l_sh + r_sh) / 2 if (l_sh is not None and r_sh is not None) else None
    mid_hip = (l_hip + r_hip) / 2 if (l_hip is not None and r_hip is not None) else None

    feat['head_turn'] = ((nose[0] - mid_sh[0]) / sw) if (nose is not None and mid_sh is not None) else 0.0
    feat['head_turn_vel'] = 0.0
    if prev_kp_frame is not None and nose is not None and mid_sh is not None:
        pv = prev_kp_frame[:, 3]
        if pv[0] >= MIN_VISIBILITY and pv[11] >= MIN_VISIBILITY and pv[12] >= MIN_VISIBILITY:
            prev_nose = prev_kp_frame[0, :2]
            prev_mid = (prev_kp_frame[11, :2] + prev_kp_frame[12, :2]) / 2
            feat['head_turn_vel'] = abs(feat['head_turn'] - (prev_nose[0] - prev_mid[0]) / sw)
    feat['nose_shoulder_dist'] = ((nose[1] - mid_sh[1]) / sw) if (nose is not None and mid_sh is not None) else 0.0
    if mid_sh is not None and mid_hip is not None:
        tv = mid_hip - mid_sh
        angle = safe_angle(tv, np.array([0, 1]))
        feat['torso_lean'] = (1.0 if tv[0] > 0 else -1.0) * angle
    else:
        feat['torso_lean'] = 0.0
    feat['shoulder_tilt'] = safe_angle(r_sh - l_sh, np.array([1, 0])) if (l_sh is not None and r_sh is not None) else 0.0
    feat['l_wrist_drop'] = ((l_wr[1] - mid_sh[1]) / sw) if (l_wr is not None and mid_sh is not None) else 0.0
    feat['r_wrist_drop'] = ((r_wr[1] - mid_sh[1]) / sw) if (r_wr is not None and mid_sh is not None) else 0.0
    feat['l_wrist_extent'] = ((l_sh[0] - l_wr[0]) / sw) if (l_wr is not None and l_sh is not None) else 0.0
    feat['r_wrist_extent'] = ((r_wr[0] - r_sh[0]) / sw) if (r_wr is not None and r_sh is not None) else 0.0
    feat['wrist_below_desk'] = 0.0
    if mid_hip is not None:
        drops = []
        if l_wr is not None: drops.append((l_wr[1] - mid_hip[1]) / sw)
        if r_wr is not None: drops.append((r_wr[1] - mid_hip[1]) / sw)
        if drops: feat['wrist_below_desk'] = max(drops)
    l_ev = vis[7] if vis[7] >= 0.1 else 0.0
    r_ev = vis[8] if vis[8] >= 0.1 else 0.0
    feat['ear_asymmetry'] = abs(l_ev - r_ev)
    feat['mouth_movement'] = 0.0
    if prev_kp_frame is not None and mouth_l is not None and mouth_r is not None:
        pv = prev_kp_frame[:, 3]
        if pv[9] >= MIN_VISIBILITY and pv[10] >= MIN_VISIBILITY:
            dl = np.linalg.norm(mouth_l - prev_kp_frame[9, :2])
            dr = np.linalg.norm(mouth_r - prev_kp_frame[10, :2])
            feat['mouth_movement'] = (dl + dr) / (2 * sw)
    feat['body_movement'] = 0.0
    if prev_kp_frame is not None:
        total, count = 0.0, 0
        for idx in [0, 11, 12, 15, 16]:
            if vis[idx] >= MIN_VISIBILITY and prev_kp_frame[idx, 3] >= MIN_VISIBILITY:
                total += np.linalg.norm(kp_frame[idx, :2] - prev_kp_frame[idx, :2])
                count += 1
        if count > 0:
            feat['body_movement'] = total / (count * sw)
    return feat


FEATURE_NAMES = [
    'head_turn', 'head_turn_vel', 'nose_shoulder_dist',
    'torso_lean', 'shoulder_tilt',
    'l_wrist_drop', 'r_wrist_drop', 'l_wrist_extent', 'r_wrist_extent',
    'wrist_below_desk', 'ear_asymmetry', 'mouth_movement', 'body_movement',
]
NUM_BASE = len(FEATURE_NAMES)
AGG_NAMES = ['mean', 'std', 'max', 'min', 'range', 'zero_crossings']
NUM_AGG = NUM_BASE * len(AGG_NAMES)


def aggregate_window(ff_arr):
    """Aggregate a window of per-frame features into a 78-dim vector."""
    if ff_arr is None or len(ff_arr) == 0:
        return np.zeros(NUM_AGG, dtype=np.float32)
    ff_arr = np.asarray(ff_arr, dtype=np.float32)
    result = []
    for i in range(ff_arr.shape[1]):
        sig = np.nan_to_num(ff_arr[:, i], nan=0.0)
        centered = sig - np.mean(sig)
        result.extend([
            float(np.mean(sig)), float(np.std(sig)),
            float(np.max(sig)), float(np.min(sig)),
            float(np.max(sig) - np.min(sig)),
            float(np.sum(np.diff(np.sign(centered)) != 0)) if len(sig) > 1 else 0.0,
        ])
    return np.array(result, dtype=np.float32)



# MODEL LOADING


class Classifier(torch.nn.Module):
    """MLP classifier — same architecture as Phase 3."""
    def __init__(self, input_dim, hidden1, hidden2, num_classes, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Linear(input_dim, hidden1), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.BatchNorm1d(hidden1),
            torch.nn.Linear(hidden1, hidden2), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden2, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def load_inference_package(pkg_path):
    """Load inference package and reconstruct models + scalers."""
    print(f'Loading inference package: {pkg_path}')
    pkg = torch.load(pkg_path, map_location='cpu', weights_only=False)
    cfg = pkg['model_config']

    model1 = Classifier(cfg['input_dim'], cfg['hidden1'], cfg['hidden2'], cfg['stage1_out'], cfg['dropout'])
    model1.load_state_dict(pkg['stage1_state_dict']); model1.eval()

    model2 = Classifier(cfg['input_dim'], cfg['hidden1'], cfg['hidden2'], cfg['stage2_out'], cfg['dropout'])
    model2.load_state_dict(pkg['stage2_state_dict']); model2.eval()

    sc1 = pickle.loads(pkg['stage1_scaler_bytes'])
    sc2 = pickle.loads(pkg['stage2_scaler_bytes'])
    threshold = pkg.get('stage1_threshold', 0.50)
    classes_s2 = pkg.get('stage2_classes', [])

    print(f'  Stage 1: threshold={threshold:.2f}')
    print(f'  Stage 2: {classes_s2}')
    return model1, model2, sc1, sc2, threshold, classes_s2, pkg


# FIX 1: BYTETRACK TRACKER (with IoU fallback)


class PersonTracker:
    """
    Multi-object tracker with two backends:
      - ByteTrack (via supervision library) — preferred, uses Kalman filter + IoU
      - IoU fallback — simple IoU-based greedy matching if supervision not installed
    """

    def __init__(self, max_people=MAX_PEOPLE):
        self.max_people = max_people
        self.tracks = {}  # track_id → {'bbox': ...}

        if HAS_SUPERVISION:
            self.backend = 'bytetrack'
            self.bt = sv.ByteTrack(
                track_activation_threshold=BYTETRACK_ACTIVATION_THRESHOLD,
                lost_track_buffer=BYTETRACK_LOST_BUFFER,
                minimum_matching_threshold=BYTETRACK_MATCHING_THRESHOLD,
                frame_rate=30,
            )
            print('  Tracker: ByteTrack (supervision)')
        else:
            self.backend = 'iou_fallback'
            self.next_id = 0
            print('  Tracker: IoU fallback')

    def update(self, detections):
        """
        Match detections to tracks. Returns list of (track_id, bbox).
        """
        if self.backend == 'bytetrack':
            return self._update_bytetrack(detections)
        else:
            return self._update_iou(detections)

    def _update_bytetrack(self, detections):
        if not detections:
            self.bt.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
        conf = np.array([d[4] for d in detections], dtype=np.float32)
        tracked = self.bt.update_with_detections(sv.Detections(xyxy=xyxy, confidence=conf))

        assignments = []
        active_ids = set()
        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                bbox = tuple(tracked.xyxy[i].astype(int))
                assignments.append((tid, bbox))
                active_ids.add(tid)
                self.tracks[tid] = {'bbox': bbox, 'missing': 0}

        for tid in list(self.tracks.keys()):
            if tid not in active_ids:
                self.tracks[tid]['missing'] = self.tracks[tid].get('missing', 0) + 1
                if self.tracks[tid]['missing'] > BYTETRACK_LOST_BUFFER:
                    del self.tracks[tid]

        return assignments[:self.max_people]

    def _update_iou(self, detections):
        """IoU-based greedy matching fallback."""
        matched_tracks = set()
        matched_dets = set()
        assignments = []
        track_ids = list(self.tracks.keys())

        if track_ids and detections:
            costs = np.zeros((len(track_ids), len(detections)))
            for i, tid in enumerate(track_ids):
                tb = self.tracks[tid]['bbox']
                for j, det in enumerate(detections):
                    db = det[:4]
                    xi1 = max(tb[0], db[0]); yi1 = max(tb[1], db[1])
                    xi2 = min(tb[2], db[2]); yi2 = min(tb[3], db[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area_t = max(1, (tb[2]-tb[0]) * (tb[3]-tb[1]))
                    area_d = max(1, (db[2]-db[0]) * (db[3]-db[1]))
                    iou = inter / (area_t + area_d - inter + 1e-6)
                    costs[i, j] = 1.0 - iou

            for _ in range(min(len(track_ids), len(detections))):
                i, j = np.unravel_index(np.argmin(costs), costs.shape)
                if costs[i, j] > IOU_MATCH_THRESHOLD:
                    break
                tid = track_ids[i]
                self.tracks[tid]['bbox'] = detections[j][:4]
                self.tracks[tid]['missing'] = 0
                matched_tracks.add(tid); matched_dets.add(j)
                assignments.append((tid, detections[j][:4]))
                costs[i, :] = float('inf'); costs[:, j] = float('inf')

        for j, det in enumerate(detections):
            if j not in matched_dets and len(self.tracks) < self.max_people:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'bbox': det[:4], 'missing': 0}
                assignments.append((tid, det[:4]))

        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid]['missing'] = self.tracks[tid].get('missing', 0) + 1
                if self.tracks[tid]['missing'] > IOU_MAX_MISSING:
                    del self.tracks[tid]

        return assignments[:self.max_people]



# PER-PERSON STATE (with FIX 2 baseline + FIX 3 flagging + FIX 4 lookdown)


class PersonState:
    """
    Per-person running state with baseline calibration, improved flagging,
    and looking-down suppression.
    """

    def __init__(self, fps, window_sec=WINDOW_SIZE_SEC, stride_sec=STRIDE_SEC):
        self.fps = fps
        self.window_frames = int(window_sec * fps)
        self.stride_frames = int(stride_sec * fps)
        self.baseline_frames = int(BASELINE_SEC * fps) if BASELINE_ENABLED else 0

        # Rolling feature buffer
        self.frame_features = deque(maxlen=self.window_frames)

        # Shoulder widths for normalisation
        self.shoulder_widths = deque(maxlen=int(5 * fps))

        # Previous frame keypoints (for velocity features)
        self.prev_kp_frame = None

        # FIX 3: Stricter flagging (4-of-7)
        self.class_history = deque(maxlen=FLAG_WINDOW)
        self.frames_since_classify = 0

        # Current state
        self.current_label = 'normal'
        self.current_prob = 0.0
        self.is_flagged = False
        self.total_frames = 0

        # FIX 2: Baseline calibration
        # Accumulate raw features during first BASELINE_SEC seconds
        self.baseline_buffer = []         # List of feature vectors during calibration
        self.baseline_mean = None         # (13,) mean of each feature during baseline
        self.is_calibrated = False        # True after calibration period ends

        # FIX 4: Looking-down suppression
        self.consecutive_looking_down = 0

    def get_median_shoulder_width(self):
        return float(np.median(list(self.shoulder_widths))) if self.shoulder_widths else 50.0

    def should_classify(self):
        return (
            len(self.frame_features) >= self.window_frames
            and self.frames_since_classify >= self.stride_frames
        )

    def get_window_array(self):
        arr = np.array(list(self.frame_features), dtype=np.float32)

        # FIX 2: If calibrated, subtract baseline mean from each frame
        if BASELINE_ENABLED and self.is_calibrated and self.baseline_mean is not None:
            arr = arr - self.baseline_mean
        return arr

    def add_frame_features(self, feat_vec):
        """
        Add a frame's feature vector and handle baseline calibration.

        Args:
            feat_vec: list of 13 floats (one per feature)
        """
        self.frame_features.append(feat_vec)
        self.frames_since_classify += 1
        self.total_frames += 1

        # FIX 2: Accumulate baseline during calibration period
        if BASELINE_ENABLED and not self.is_calibrated:
            self.baseline_buffer.append(feat_vec)
            if self.total_frames >= self.baseline_frames and len(self.baseline_buffer) > 10:
                # Calibration complete — compute baseline mean
                self.baseline_mean = np.mean(self.baseline_buffer, axis=0).astype(np.float32)
                self.is_calibrated = True
                self.baseline_buffer = []  # Free memory

    @property
    def is_in_calibration(self):
        """True if still in the baseline calibration period."""
        return BASELINE_ENABLED and not self.is_calibrated

    def update_classification(self, label, prob):
        """
        Record classification result with looking-down suppression and flagging.
        """
        # FIX 4: Looking-down suppression
        if LOOKDOWN_SUPPRESS and label == 'looking_down':
            self.consecutive_looking_down += 1
            if self.consecutive_looking_down > LOOKDOWN_MAX_CONSECUTIVE:
                label = 'normal'
                prob = 1.0 - prob
        else:
            self.consecutive_looking_down = 0

        self.current_label = label
        self.current_prob = prob
        self.class_history.append(1 if label != 'normal' else 0)
        self.frames_since_classify = 0

        # FIX 3: Stricter flagging (4-of-7)
        if len(self.class_history) >= FLAG_THRESHOLD:
            self.is_flagged = sum(self.class_history) >= FLAG_THRESHOLD
        else:
            self.is_flagged = False



# INFERENCE ENGINE


class InferenceEngine:
    """Full pipeline: detection → tracking → pose → features → classification."""

    def __init__(self, pkg_path):
        self.model1, self.model2, self.sc1, self.sc2, self.threshold, \
            self.classes_s2, self.config = load_inference_package(pkg_path)

        print(f'Loading YOLOv8 ({YOLO_MODEL})...')
        self.yolo = YOLO(YOLO_MODEL)

        print('Initialising MediaPipe Pose...')
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )

        self.tracker = PersonTracker()
        self.person_states = {}

        print(f'  Baseline calibration: {"ON" if BASELINE_ENABLED else "OFF"} ({BASELINE_SEC}s)')
        print(f'  Flagging rule: {FLAG_THRESHOLD}-of-{FLAG_WINDOW}')
        print(f'  Looking-down suppression: {"ON" if LOOKDOWN_SUPPRESS else "OFF"}')
        print('Inference engine ready.\n')

    def detect_people(self, frame):
        results = self.yolo(frame, verbose=False, classes=[PERSON_CLASS_ID], conf=YOLO_CONF)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        boxes.sort(key=lambda b: b[0])
        return boxes[:MAX_PEOPLE]

    def extract_pose(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.pose.process(crop_rgb)
        kp = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks.landmark):
                kp[i] = [lm.x, lm.y, lm.z, lm.visibility]
        return kp

    def classify_window(self, window_arr):
        """Two-stage classification: normal/suspicious → suspicious subtype."""
        agg = aggregate_window(window_arr)
        agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

        x1 = self.sc1.transform(agg).astype(np.float32)
        with torch.no_grad():
            probs1 = torch.softmax(self.model1(torch.FloatTensor(x1)), dim=1).numpy()[0]
        p_sus = float(probs1[1])

        if p_sus < self.threshold:
            return 'normal', 1.0 - p_sus

        x2 = self.sc2.transform(agg).astype(np.float32)
        with torch.no_grad():
            probs2 = torch.softmax(self.model2(torch.FloatTensor(x2)), dim=1).numpy()[0]
        class_idx = int(np.argmax(probs2))
        return self.classes_s2[class_idx], float(probs2[class_idx])

    def process_frame(self, frame, fps):
        """Process one frame. Returns list of per-person result dicts."""
        detections = self.detect_people(frame)
        assignments = self.tracker.update(detections)
        results = []

        for track_id, bbox in assignments:
            if track_id not in self.person_states:
                self.person_states[track_id] = PersonState(fps)
            ps = self.person_states[track_id]

            kp_crop = self.extract_pose(frame, bbox)
            kp_frame = kp_to_frame_coords(kp_crop, bbox)

            vis = kp_frame[:, 3]
            if vis[11] >= MIN_VISIBILITY and vis[12] >= MIN_VISIBILITY:
                sw = np.linalg.norm(kp_frame[12, :2] - kp_frame[11, :2])
                if sw > 5:
                    ps.shoulder_widths.append(sw)

            ref_sw = ps.get_median_shoulder_width()
            feat = compute_frame_features(kp_frame, ps.prev_kp_frame, ref_sw)
            ps.prev_kp_frame = kp_frame.copy()

            feat_vec = [feat[fn] for fn in FEATURE_NAMES]
            ps.add_frame_features(feat_vec)

            # Classify (skip during calibration period)
            if ps.should_classify() and not ps.is_in_calibration:
                window_arr = ps.get_window_array()
                label, prob = self.classify_window(window_arr)
                ps.update_classification(label, prob)

            is_new = ps.total_frames < ps.window_frames
            results.append({
                'track_id': track_id,
                'bbox': bbox,
                'label': ps.current_label,
                'prob': ps.current_prob,
                'is_flagged': ps.is_flagged,
                'is_new': is_new,
                'is_calibrating': ps.is_in_calibration,
            })

        active_ids = {tid for tid, _ in assignments}
        for tid in list(self.person_states.keys()):
            if tid not in active_ids and tid not in self.tracker.tracks:
                del self.person_states[tid]

        return results

    def draw_annotations(self, frame, results, frame_idx, fps, total_frames=None):
        """Draw bounding boxes, labels, and status bar."""
        h, w = frame.shape[:2]
        out = frame.copy()

        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r['bbox']]
            label = r['label']
            prob = r['prob']
            flagged = r['is_flagged']
            is_new = r['is_new']
            calibrating = r.get('is_calibrating', False)

            if calibrating:
                color = COLOR_CALIBRATING
                status_text = 'calibrating...'
            elif is_new:
                color = COLOR_UNTRACKED
                status_text = 'warming up...'
            elif flagged:
                color = COLOR_SUSPICIOUS
                status_text = f'FLAGGED: {label} ({prob:.0%})'
            elif label != 'normal':
                color = COLOR_WARNING
                status_text = f'{label} ({prob:.0%})'
            else:
                color = COLOR_NORMAL
                status_text = f'normal ({prob:.0%})'

            thickness = 3 if flagged else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            label_text = f'P{r["track_id"]+1}: {status_text}'
            (tw, th_t), _ = cv2.getTextSize(label_text, FONT, 0.50, 1)
            ty = max(th_t + 10, y1 - 8)
            cv2.rectangle(out, (x1, ty - th_t - 4), (x1 + tw + 8, ty + 4), (30, 30, 30), -1)
            cv2.putText(out, label_text, (x1 + 4, ty), FONT, 0.50, color, 1)

            if flagged:
                cv2.rectangle(out, (x1, y2), (x2, y2 + 6), COLOR_SUSPICIOUS, -1)

        cv2.rectangle(out, (0, 0), (w, 36), (20, 20, 20), -1)
        time_sec = frame_idx / fps if fps > 0 else 0
        n_flagged = sum(1 for r in results if r['is_flagged'])
        n_sus = sum(1 for r in results if r['label'] != 'normal' and not r['is_new'])
        info = f'Frame {frame_idx}'
        if total_frames: info += f'/{total_frames}'
        info += f'  |  Time: {time_sec:.1f}s  |  People: {len(results)}'
        info += f'  |  Suspicious: {n_sus}  |  Flagged: {n_flagged}'
        info += f'  |  Thr: {self.threshold:.2f}  |  Rule: {FLAG_THRESHOLD}/{FLAG_WINDOW}'
        cv2.putText(out, info, (10, 24), FONT, 0.48, (255, 255, 255), 1)

        return out



# VIDEO PROCESSING
def process_video(engine, video_path, output_dir, show=False):
    """Process a video file and produce annotated output + timeline CSV + summary."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'ERROR: Could not open {video_path}'); return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0
    stem = video_path.stem

    print(f'Processing: {video_path.name}')
    print(f'  {width}x{height} @ {fps:.1f} FPS, {duration_sec:.1f}s ({total_frames} frames)')

    out_video_path = output_dir / f'{stem}_annotated.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    timeline_path = output_dir / f'{stem}_timeline.csv'
    timeline_rows = []

    # Reset engine state
    engine.tracker = PersonTracker()
    engine.person_states = {}

    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = engine.process_frame(frame, fps)
        annotated = engine.draw_annotations(frame, results, frame_idx, fps, total_frames)
        writer.write(annotated)

        if show:
            preview = cv2.resize(annotated, (1280, 720)) if width > 1280 else annotated
            cv2.imshow(f'Phase 4 v2 — {stem}', preview)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                print('  Stopped by user.'); break

        if frame_idx % max(1, int(fps)) == 0:
            time_sec = frame_idx / fps
            for r in results:
                timeline_rows.append({
                    'time_sec': round(time_sec, 1), 'frame': frame_idx,
                    'track_id': r['track_id'], 'label': r['label'],
                    'prob': round(r['prob'], 3), 'is_flagged': r['is_flagged'],
                    'bbox_x1': int(r['bbox'][0]), 'bbox_y1': int(r['bbox'][1]),
                    'bbox_x2': int(r['bbox'][2]), 'bbox_y2': int(r['bbox'][3]),
                })

        if (frame_idx + 1) % (int(fps) * 10) == 0:
            elapsed = time.time() - start_time
            pct = (frame_idx + 1) / total_frames * 100
            fps_proc = (frame_idx + 1) / elapsed
            eta = (total_frames - frame_idx - 1) / fps_proc / 60
            print(f'  [{pct:5.1f}%] frame {frame_idx+1}/{total_frames}  '
                  f'{fps_proc:.1f} fps  ETA: {eta:.1f} min')

        frame_idx += 1

    cap.release(); writer.release()
    if show: cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps_proc = frame_idx / elapsed if elapsed > 0 else 0

    if timeline_rows:
        with open(timeline_path, 'w', newline='', encoding='utf-8') as f:
            fns = ['time_sec','frame','track_id','label','prob','is_flagged',
                   'bbox_x1','bbox_y1','bbox_x2','bbox_y2']
            w = csv.DictWriter(f, fieldnames=fns)
            w.writeheader(); w.writerows(timeline_rows)

    summary_path = output_dir / f'{stem}_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'Phase 4 v2 Inference Summary\n{"="*30}\n\n')
        f.write(f'Video: {video_path.name}\n')
        f.write(f'Duration: {duration_sec:.1f} seconds\n')
        f.write(f'Frames: {frame_idx}\n')
        f.write(f'Speed: {fps_proc:.1f} fps\n')
        f.write(f'Time: {elapsed:.1f} seconds\n')
        f.write(f'Threshold: {engine.threshold:.2f}\n')
        f.write(f'Flagging rule: {FLAG_THRESHOLD}-of-{FLAG_WINDOW}\n')
        f.write(f'Baseline calibration: {BASELINE_SEC}s\n')
        f.write(f'Tracker: {"ByteTrack" if HAS_SUPERVISION else "IoU fallback"}\n')

    print(f'\n  DONE: {video_path.name}')
    print(f'  Video: {out_video_path}')
    print(f'  Timeline: {timeline_path}')
    print(f'  Speed: {fps_proc:.1f} fps ({elapsed:.1f} sec)')



# MAIN

def main():
    parser = argparse.ArgumentParser(
        description='Phase 4 v2: Improved Live Inference Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python phase4_inference_v2.py --video exam.mp4 --pkg inference_package_v11.pth
    python phase4_inference_v2.py --video exam.mp4 --pkg inference_package_v11.pth --show
    python phase4_inference_v2.py --video_dir 03_long_test_videos/ --pkg inference_package_v11.pth
    python phase4_inference_v2.py --webcam --pkg inference_package_v11.pth
        """)
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--video_dir', type=str, help='Directory of video files')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--pkg', type=str, default=DEFAULT_INFERENCE_PKG, help='Inference package path')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show live preview')
    parser.add_argument('--threshold', type=float, default=None, help='Override Stage 1 threshold')
    parser.add_argument('--no-baseline', action='store_true', help='Disable baseline calibration')

    args = parser.parse_args()

    if not args.video and not args.video_dir and not args.webcam:
        parser.error('Provide --video, --video_dir, or --webcam')

    if not os.path.exists(args.pkg):
        print(f'ERROR: Inference package not found: {args.pkg}')
        sys.exit(1)

    if args.no_baseline:
        global BASELINE_ENABLED
        BASELINE_ENABLED = False

    engine = InferenceEngine(args.pkg)
    if args.threshold is not None:
        engine.threshold = args.threshold
        print(f'  Threshold overridden: {args.threshold:.2f}')

    if args.video:
        if not os.path.exists(args.video):
            print(f'ERROR: Video not found: {args.video}'); sys.exit(1)
        process_video(engine, args.video, args.output, show=args.show)
    elif args.video_dir:
        vdir = Path(args.video_dir)
        videos = sorted(vdir.glob('*.mp4'))
        if not videos:
            print(f'No .mp4 files in {vdir}'); sys.exit(1)
        print(f'Found {len(videos)} videos\n')
        for vp in videos:
            process_video(engine, vp, args.output, show=args.show); print()
    elif args.webcam:
        print('Webcam mode — press Q to quit')
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('ERROR: Could not open webcam'); sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        engine.tracker = PersonTracker()
        engine.person_states = {}
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            results = engine.process_frame(frame, fps)
            annotated = engine.draw_annotations(frame, results, frame_idx, fps)
            cv2.imshow('Phase 4 v2 — Webcam', annotated)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: break
            frame_idx += 1
        cap.release(); cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        import pandas as pd
    except ImportError:
        pd = None
    main()
