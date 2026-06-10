"""
--- PHASE 4: Live Inference Pipeline ---
Exam Proctoring — Pose-Based Suspicious Behaviour Detection

This script processes a video file (or webcam feed) and produces an annotated output video with colour-coded bounding boxes:
    GREEN  = normal behaviour
    RED    = suspicious behaviour (with class label)
    YELLOW = suspicious but below flagging threshold (early warning)

PIPELINE PER FRAME:
  1. YOLOv8 detects all people → bounding boxes
  2. MediaPipe extracts 33 keypoints per person (cropped from bbox)
  3. Per-person feature buffers accumulate frame-level features
  4. Every STRIDE_SEC seconds, a 3-second window is classified:
       Stage 1: normal vs suspicious (with probability threshold)
       Stage 2: if suspicious → head_turn / lateral_movement / looking_down
  5. Temporal smoothing: 3-of-5 flagging rule over a 10-second sliding window
  6. Draw coloured bounding boxes on the frame

"""

import os
import sys
import csv
import time
import pickle
import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import cv2
import torch
import mediapipe as mp
from ultralytics import YOLO


# --- CONFIGURATION ---


# --- Model paths ---
# Update this to point to your inference package from Phase 3 v9
DEFAULT_INFERENCE_PKG = './inference_package_v9.pth'

# --- YOLOv8 ---
YOLO_MODEL       = 'yolov8n.pt'   # Nano model — fast on CPU
YOLO_CONF        = 0.5            # Confidence threshold for person detection
PERSON_CLASS_ID  = 0              # COCO class ID for "person"
MAX_PEOPLE       = 8              # Maximum people to track per frame

# --- MediaPipe Pose ---
MP_MODEL_COMPLEXITY      = 1      # 0=lite, 1=full, 2=heavy
MP_MIN_DETECTION_CONF    = 0.5
MP_MIN_TRACKING_CONF     = 0.5

# --- Feature computation ---
MIN_VISIBILITY   = 0.5            # MediaPipe keypoint visibility threshold
NUM_KEYPOINTS    = 33             # MediaPipe pose landmarks

# --- Sliding window ---
WINDOW_SIZE_SEC  = 3.0            # Each classification window covers 3 seconds
STRIDE_SEC       = 1.0            # Classify every 1 second

# --- Temporal smoothing (flagging rule) ---
# A person is "flagged" (solid red) only if they have been classified as suspicious in at least FLAG_THRESHOLD out of the last FLAG_WINDOW recent classification results. This prevents single-frame false alarms.
FLAG_WINDOW      = 5              # Look at the last 5 classification results
FLAG_THRESHOLD   = 3              # Need 3 out of 5 to be suspicious → flagged

# --- Visualisation ---
COLOR_NORMAL     = (0, 200, 0)    # Green (BGR)
COLOR_SUSPICIOUS = (0, 0, 220)    # Red (BGR)
COLOR_WARNING    = (0, 200, 255)  # Yellow - suspicious but not yet flagged
COLOR_UNTRACKED  = (180, 180, 180)  # Grey - person detected but no classification yet
FONT             = cv2.FONT_HERSHEY_SIMPLEX


# --- FEATURE COMPUTATION ---


def kp_to_frame_coords(kp, bbox):
    """ Convert crop-normalised keypoints (0-1) back to frame pixel coordinates using the bounding box that was used to create the crop. """
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
    """ Compute 13 pose features from one frame's keypoints in frame coordinates. """
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

    # F1: Head turn
    feat['head_turn'] = ((nose[0] - mid_sh[0]) / sw) if (nose is not None and mid_sh is not None) else 0.0

    # F2: Head turn velocity
    feat['head_turn_vel'] = 0.0
    if prev_kp_frame is not None and nose is not None and mid_sh is not None:
        pv = prev_kp_frame[:, 3]
        if pv[0] >= MIN_VISIBILITY and pv[11] >= MIN_VISIBILITY and pv[12] >= MIN_VISIBILITY:
            prev_nose = prev_kp_frame[0, :2]
            prev_mid = (prev_kp_frame[11, :2] + prev_kp_frame[12, :2]) / 2
            feat['head_turn_vel'] = abs(feat['head_turn'] - (prev_nose[0] - prev_mid[0]) / sw)

    # F3: Nose-shoulder distance
    feat['nose_shoulder_dist'] = ((nose[1] - mid_sh[1]) / sw) if (nose is not None and mid_sh is not None) else 0.0

    # F4: Torso lean
    if mid_sh is not None and mid_hip is not None:
        tv = mid_hip - mid_sh
        angle = safe_angle(tv, np.array([0, 1]))
        feat['torso_lean'] = (1.0 if tv[0] > 0 else -1.0) * angle
    else:
        feat['torso_lean'] = 0.0

    # F5: Shoulder tilt
    feat['shoulder_tilt'] = safe_angle(r_sh - l_sh, np.array([1, 0])) if (l_sh is not None and r_sh is not None) else 0.0

    # F6-F7: Wrist drop
    feat['l_wrist_drop'] = ((l_wr[1] - mid_sh[1]) / sw) if (l_wr is not None and mid_sh is not None) else 0.0
    feat['r_wrist_drop'] = ((r_wr[1] - mid_sh[1]) / sw) if (r_wr is not None and mid_sh is not None) else 0.0

    # F8-F9: Wrist extent
    feat['l_wrist_extent'] = ((l_sh[0] - l_wr[0]) / sw) if (l_wr is not None and l_sh is not None) else 0.0
    feat['r_wrist_extent'] = ((r_wr[0] - r_sh[0]) / sw) if (r_wr is not None and r_sh is not None) else 0.0

    # F10: Wrist below desk
    feat['wrist_below_desk'] = 0.0
    if mid_hip is not None:
        drops = []
        if l_wr is not None: drops.append((l_wr[1] - mid_hip[1]) / sw)
        if r_wr is not None: drops.append((r_wr[1] - mid_hip[1]) / sw)
        if drops: feat['wrist_below_desk'] = max(drops)

    # F11: Ear asymmetry
    l_ev = vis[7] if vis[7] >= 0.1 else 0.0
    r_ev = vis[8] if vis[8] >= 0.1 else 0.0
    feat['ear_asymmetry'] = abs(l_ev - r_ev)

    # F12: Mouth movement
    feat['mouth_movement'] = 0.0
    if prev_kp_frame is not None and mouth_l is not None and mouth_r is not None:
        pv = prev_kp_frame[:, 3]
        if pv[9] >= MIN_VISIBILITY and pv[10] >= MIN_VISIBILITY:
            dl = np.linalg.norm(mouth_l - prev_kp_frame[9, :2])
            dr = np.linalg.norm(mouth_r - prev_kp_frame[10, :2])
            feat['mouth_movement'] = (dl + dr) / (2 * sw)

    # F13: Body movement
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


# Canonical feature order must match the order used in training Phase 3's models!
FEATURE_NAMES = [
    'head_turn', 'head_turn_vel', 'nose_shoulder_dist',
    'torso_lean', 'shoulder_tilt',
    'l_wrist_drop', 'r_wrist_drop', 'l_wrist_extent', 'r_wrist_extent',
    'wrist_below_desk', 'ear_asymmetry', 'mouth_movement', 'body_movement',
]
NUM_BASE = len(FEATURE_NAMES)
AGG_NAMES = ['mean', 'std', 'max', 'min', 'range', 'zero_crossings']
NUM_AGG = NUM_BASE * len(AGG_NAMES)  # 78


def aggregate_window(ff_arr):
    """ Aggregate a window of per-frame features into a 78-dim vector. Identical to Phase 2's aggregation."""
    if ff_arr is None or len(ff_arr) == 0:
        return np.zeros(NUM_AGG, dtype=np.float32)
    ff_arr = np.asarray(ff_arr, dtype=np.float32)
    result = []
    for i in range(ff_arr.shape[1]):
        sig = np.nan_to_num(ff_arr[:, i], nan=0.0)
        centered = sig - np.mean(sig)
        result.extend([
            float(np.mean(sig)),
            float(np.std(sig)),
            float(np.max(sig)),
            float(np.min(sig)),
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
            torch.nn.Linear(input_dim, hidden1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm1d(hidden1),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden2, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def load_inference_package(pkg_path):
    """
    Load the Phase 3 inference package and reconstruct both models + scalers.

    Returns:
        model1:     Stage 1 binary classifier (normal vs suspicious)
        model2:     Stage 2 merged subtype classifier (3-class)
        sc1:        StandardScaler for Stage 1
        sc2:        StandardScaler for Stage 2
        threshold:  Optimal Stage 1 probability threshold
        classes_s2: Stage 2 class names (e.g., ['head_turn', 'lateral_movement', 'looking_down'])
        config:     Full package dict for reference
    """
    print(f'Loading inference package: {pkg_path}')
    pkg = torch.load(pkg_path, map_location='cpu', weights_only=False)

    cfg = pkg['model_config']

    # Reconstruct Stage 1 (binary: normal vs suspicious)
    model1 = Classifier(cfg['input_dim'], cfg['hidden1'], cfg['hidden2'],
                        cfg['stage1_out'], cfg['dropout'])
    model1.load_state_dict(pkg['stage1_state_dict'])
    model1.eval()

    # Reconstruct Stage 2 (3-class merged suspicious subtype)
    model2 = Classifier(cfg['input_dim'], cfg['hidden1'], cfg['hidden2'],
                        cfg['stage2_out'], cfg['dropout'])
    model2.load_state_dict(pkg['stage2_state_dict'])
    model2.eval()

    # Reconstruct scalers
    sc1 = pickle.loads(pkg['stage1_scaler_bytes'])
    sc2 = pickle.loads(pkg['stage2_scaler_bytes'])

    # Threshold
    threshold = pkg.get('stage1_threshold', 0.50)

    # Class names
    classes_s2 = pkg.get('stage2_classes', ['head_turn', 'lateral_movement', 'looking_down'])
    merge_rev = pkg.get('merge_reverse', {})

    print(f'  Stage 1: normal vs suspicious (threshold={threshold:.2f})')
    print(f'  Stage 2: {classes_s2}')
    print(f'  Merge mapping: {merge_rev}')
    print(f'  Test macro F1: {pkg.get("test_macro_f1_optimal", "N/A")}')

    return model1, model2, sc1, sc2, threshold, classes_s2, pkg



# --- PERSON TRACKER ---


class PersonTracker:
    """
    Tracks people across frames using spatial proximity.

    Since the camera is fixed and people are seated, we use a simple
    nearest-neighbour assignment based on bounding box centre distance.
    Each tracked person gets a stable ID that persists across frames.

    This is simpler and more reliable than DeepSORT/ByteTrack for our
    fixed-camera, seated-people setup.
    """

    def __init__(self, max_people=MAX_PEOPLE, max_dist=150, max_missing=30):
        """
        Args:
            max_people:  Maximum number of people to track simultaneously
            max_dist:    Maximum pixel distance to match a detection to a track
            max_missing: Frames before a lost track is removed
        """
        self.max_people = max_people
        self.max_dist = max_dist
        self.max_missing = max_missing
        self.tracks = {}        # track_id → {'cx', 'cy', 'bbox', 'missing', 'feature_buffer', ...}
        self.next_id = 0

    def _centre(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections):
        """ Match new detections to existing tracks. """
        det_centres = [self._centre(d[:4]) for d in detections]

        # Build cost matrix (distance between each track and each detection)
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()
        assignments = []

        if track_ids and detections:
            costs = np.zeros((len(track_ids), len(detections)))
            for i, tid in enumerate(track_ids):
                tc = (self.tracks[tid]['cx'], self.tracks[tid]['cy'])
                for j, dc in enumerate(det_centres):
                    costs[i, j] = np.sqrt((tc[0] - dc[0])**2 + (tc[1] - dc[1])**2)

            # Greedy matching (good enough for 8 seated people)
            for _ in range(min(len(track_ids), len(detections))):
                i, j = np.unravel_index(np.argmin(costs), costs.shape)
                if costs[i, j] > self.max_dist:
                    break
                tid = track_ids[i]
                self.tracks[tid]['cx'] = det_centres[j][0]
                self.tracks[tid]['cy'] = det_centres[j][1]
                self.tracks[tid]['bbox'] = detections[j][:4]
                self.tracks[tid]['missing'] = 0
                matched_tracks.add(tid)
                matched_dets.add(j)
                assignments.append((tid, detections[j][:4]))
                costs[i, :] = float('inf')
                costs[:, j] = float('inf')

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets and len(self.tracks) < self.max_people:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'cx': det_centres[j][0],
                    'cy': det_centres[j][1],
                    'bbox': det[:4],
                    'missing': 0,
                }
                matched_tracks.add(tid)
                assignments.append((tid, det[:4]))

        # Increment missing counter for unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid]['missing'] += 1
                if self.tracks[tid]['missing'] > self.max_missing:
                    del self.tracks[tid]

        return assignments


# PER-PERSON STATE (feature buffer + classification history)


class PersonState:
    """
    Maintains the running state for one tracked person:
    - Rolling buffer of per-frame features (for window aggregation)
    - Recent shoulder widths (for median computation)
    - Previous frame keypoints (for velocity features)
    - Classification history (for temporal smoothing / flagging rule)
    """

    def __init__(self, fps, window_sec=WINDOW_SIZE_SEC, stride_sec=STRIDE_SEC):
        self.fps = fps
        self.window_frames = int(window_sec * fps)
        self.stride_frames = int(stride_sec * fps)

        # Rolling feature buffer (keeps last window_frames of per-frame features)
        self.frame_features = deque(maxlen=self.window_frames)

        # Shoulder widths for median computation
        self.shoulder_widths = deque(maxlen=int(5 * fps))  # Last 5 seconds

        # Previous frame keypoints (for velocity features)
        self.prev_kp_frame = None

        # Classification history (for flagging rule)
        self.class_history = deque(maxlen=FLAG_WINDOW)

        # Frame counter since last classification
        self.frames_since_classify = 0

        # Current state
        self.current_label = 'normal'
        self.current_prob = 0.0
        self.is_flagged = False   # True when 3-of-5 rule triggers
        self.total_frames = 0

    def get_median_shoulder_width(self):
        """Median shoulder width in pixels. Fallback: 50.0."""
        if self.shoulder_widths:
            return float(np.median(list(self.shoulder_widths)))
        return 50.0

    def should_classify(self):
        """True if we have enough frames AND it's time for a new classification."""
        return (
            len(self.frame_features) >= self.window_frames
            and self.frames_since_classify >= self.stride_frames
        )

    def get_window_array(self):
        """Return the current window as a (window_frames, 13) numpy array."""
        return np.array(list(self.frame_features), dtype=np.float32)

    def update_classification(self, label, prob):
        """Record a new classification result and apply flagging rule."""
        self.current_label = label
        self.current_prob = prob
        self.class_history.append(1 if label != 'normal' else 0)
        self.frames_since_classify = 0

        # 3-of-5 flagging rule
        if len(self.class_history) >= FLAG_THRESHOLD:
            recent_sus = sum(self.class_history)
            self.is_flagged = recent_sus >= FLAG_THRESHOLD



# MAIN INFERENCE ENGINE


class InferenceEngine:
    """ Orchestrates the full pipeline: detection → pose → features → classification. """

    def __init__(self, pkg_path):
        # Load models
        self.model1, self.model2, self.sc1, self.sc2, self.threshold, \
            self.classes_s2, self.config = load_inference_package(pkg_path)

        # Load YOLOv8
        print(f'Loading YOLOv8 ({YOLO_MODEL})...')
        self.yolo = YOLO(YOLO_MODEL)

        # Initialise MediaPipe Pose
        print('Initialising MediaPipe Pose...')
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )

        # Tracker
        self.tracker = PersonTracker()

        # Per-person state
        self.person_states = {}  # track_id → PersonState

        print('Inference engine ready.\n')

    def detect_people(self, frame):
        """Run YOLOv8 to get person bounding boxes, sorted left-to-right."""
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
        """Extract MediaPipe keypoints from a cropped person region."""
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
        """ Run the two-stage classifier on an aggregated feature vector. """
        agg = aggregate_window(window_arr)
        agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

        # Stage 1: normal vs suspicious
        x1 = self.sc1.transform(agg).astype(np.float32)
        with torch.no_grad():
            logits1 = self.model1(torch.FloatTensor(x1))
            probs1 = torch.softmax(logits1, dim=1).numpy()[0]

        p_sus = float(probs1[1])

        if p_sus < self.threshold:
            return 'normal', 1.0 - p_sus

        # Stage 2: which type of suspicious?
        x2 = self.sc2.transform(agg).astype(np.float32)
        with torch.no_grad():
            logits2 = self.model2(torch.FloatTensor(x2))
            probs2 = torch.softmax(logits2, dim=1).numpy()[0]

        class_idx = int(np.argmax(probs2))
        label = self.classes_s2[class_idx]
        prob = float(probs2[class_idx])

        return label, prob

    def process_frame(self, frame, fps):
        """ Process one video frame through the full pipeline. """
        # Step 1: Detect people
        detections = self.detect_people(frame)

        # Step 2: Track across frames
        assignments = self.tracker.update(detections)

        results = []

        for track_id, bbox in assignments:
            # Ensure PersonState exists for this track
            if track_id not in self.person_states:
                self.person_states[track_id] = PersonState(fps)
            ps = self.person_states[track_id]

            # Step 3: Extract pose keypoints
            kp_crop = self.extract_pose(frame, bbox)

            # Convert to frame coordinates
            kp_frame = kp_to_frame_coords(kp_crop, bbox)

            # Update shoulder width estimate
            vis = kp_frame[:, 3]
            if vis[11] >= MIN_VISIBILITY and vis[12] >= MIN_VISIBILITY:
                sw = np.linalg.norm(kp_frame[12, :2] - kp_frame[11, :2])
                if sw > 5:
                    ps.shoulder_widths.append(sw)

            # Step 4: Compute per-frame features
            ref_sw = ps.get_median_shoulder_width()
            feat = compute_frame_features(kp_frame, ps.prev_kp_frame, ref_sw)
            ps.prev_kp_frame = kp_frame.copy()

            # Add to rolling buffer
            feat_vec = [feat[fn] for fn in FEATURE_NAMES]
            ps.frame_features.append(feat_vec)
            ps.frames_since_classify += 1
            ps.total_frames += 1

            # Step 5: Classify if window is ready
            if ps.should_classify():
                window_arr = ps.get_window_array()
                label, prob = self.classify_window(window_arr)
                ps.update_classification(label, prob)

            # Build result for this person
            is_new = ps.total_frames < ps.window_frames
            results.append({
                'track_id': track_id,
                'bbox': bbox,
                'label': ps.current_label,
                'prob': ps.current_prob,
                'is_flagged': ps.is_flagged,
                'is_new': is_new,
            })

        # Clean up states for tracks that no longer exist
        active_ids = {tid for tid, _ in assignments}
        for tid in list(self.person_states.keys()):
            if tid not in active_ids and tid not in self.tracker.tracks:
                del self.person_states[tid]

        return results

    def draw_annotations(self, frame, results, frame_idx, fps, total_frames=None):
        """Draw bounding boxes, labels, and status bar on the frame."""
        h, w = frame.shape[:2]
        out = frame.copy()

        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r['bbox']]
            label = r['label']
            prob = r['prob']
            flagged = r['is_flagged']
            is_new = r['is_new']

            # Determine colour
            if is_new:
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

            # Draw bounding box
            thickness = 3 if flagged else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            label_text = f'P{r["track_id"]+1}: {status_text}'
            (tw, th), baseline = cv2.getTextSize(label_text, FONT, 0.50, 1)
            ty = max(th + 10, y1 - 8)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 8, ty + 4), (30, 30, 30), -1)
            cv2.putText(out, label_text, (x1 + 4, ty), FONT, 0.50, color, 1)

            # Draw flag indicator
            if flagged:
                cv2.rectangle(out, (x1, y2), (x2, y2 + 6), COLOR_SUSPICIOUS, -1)

        # Top info bar
        cv2.rectangle(out, (0, 0), (w, 36), (20, 20, 20), -1)
        time_sec = frame_idx / fps if fps > 0 else 0
        n_flagged = sum(1 for r in results if r['is_flagged'])
        n_sus = sum(1 for r in results if r['label'] != 'normal' and not r['is_new'])
        info = f'Frame {frame_idx}'
        if total_frames:
            info += f'/{total_frames}'
        info += f'  |  Time: {time_sec:.1f}s  |  People: {len(results)}'
        info += f'  |  Suspicious: {n_sus}  |  Flagged: {n_flagged}'
        info += f'  |  Threshold: {self.threshold:.2f}'
        cv2.putText(out, info, (10, 24), FONT, 0.50, (255, 255, 255), 1)

        return out



# VIDEO PROCESSING


def process_video(engine, video_path, output_dir, show=False):
    """
    Process a complete video file through the inference pipeline.

    Produces:
      1. Annotated video with coloured bounding boxes
      2. Timeline CSV (per-person, per-second classification log)
      3. Summary text file
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'ERROR: Could not open {video_path}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    stem = video_path.stem

    print(f'Processing: {video_path.name}')
    print(f'  Resolution: {width}x{height} @ {fps:.1f} FPS')
    print(f'  Duration: {duration_sec:.1f} seconds ({total_frames} frames)')

    # Output video writer
    out_video_path = output_dir / f'{stem}_annotated.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # Timeline log
    timeline_path = output_dir / f'{stem}_timeline.csv'
    timeline_rows = []

    # Reset engine state for new video
    engine.tracker = PersonTracker()
    engine.person_states = {}

    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = engine.process_frame(frame, fps)

        # Draw annotations
        annotated = engine.draw_annotations(frame, results, frame_idx, fps, total_frames)

        # Write output frame
        writer.write(annotated)

        # Show live preview
        if show:
            preview = cv2.resize(annotated, (1280, 720)) if width > 1280 else annotated
            cv2.imshow(f'Phase 4 — {stem}', preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print('  Stopped by user.')
                break

        # Log timeline (every second)
        if frame_idx % int(fps) == 0:
            time_sec = frame_idx / fps
            for r in results:
                timeline_rows.append({
                    'time_sec': round(time_sec, 1),
                    'frame': frame_idx,
                    'track_id': r['track_id'],
                    'label': r['label'],
                    'prob': round(r['prob'], 3),
                    'is_flagged': r['is_flagged'],
                    'bbox_x1': int(r['bbox'][0]),
                    'bbox_y1': int(r['bbox'][1]),
                    'bbox_x2': int(r['bbox'][2]),
                    'bbox_y2': int(r['bbox'][3]),
                })

        # Progress
        if (frame_idx + 1) % (int(fps) * 10) == 0:
            elapsed = time.time() - start_time
            pct = (frame_idx + 1) / total_frames * 100
            fps_proc = (frame_idx + 1) / elapsed
            eta = (total_frames - frame_idx - 1) / fps_proc / 60
            print(f'  [{pct:5.1f}%] frame {frame_idx+1}/{total_frames}  '
                  f'{fps_proc:.1f} fps  ETA: {eta:.1f} min')

        frame_idx += 1

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps_proc = frame_idx / elapsed if elapsed > 0 else 0

    # Save timeline CSV
    if timeline_rows:
        with open(timeline_path, 'w', newline='') as f:
            fieldnames = ['time_sec', 'frame', 'track_id', 'label', 'prob',
                          'is_flagged', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(timeline_rows)

    # Generate summary
    summary_path = output_dir / f'{stem}_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f'Phase 4 Inference Summary\n')
        f.write(f'========================\n\n')
        f.write(f'Video: {video_path.name}\n')
        f.write(f'Duration: {duration_sec:.1f} seconds\n')
        f.write(f'Frames processed: {frame_idx}\n')
        f.write(f'Processing speed: {fps_proc:.1f} fps\n')
        f.write(f'Processing time: {elapsed:.1f} seconds\n')
        f.write(f'Threshold: {engine.threshold:.2f}\n\n')

        # Per-person summary
        if timeline_rows:
            df_tl = pd.DataFrame(timeline_rows) if 'pd' in dir() else None
            if df_tl is not None:
                import pandas as pd
                df_tl = pd.DataFrame(timeline_rows)
                f.write('Per-Person Summary:\n')
                f.write('-' * 40 + '\n')
                for tid in sorted(df_tl['track_id'].unique()):
                    person_df = df_tl[df_tl['track_id'] == tid]
                    total_entries = len(person_df)
                    sus_entries = (person_df['label'] != 'normal').sum()
                    flagged_entries = person_df['is_flagged'].sum()
                    labels = person_df['label'].value_counts()
                    f.write(f'\n  Person {tid+1} (P{tid+1}):\n')
                    f.write(f'    Observations: {total_entries}\n')
                    f.write(f'    Suspicious: {sus_entries} ({sus_entries/max(total_entries,1)*100:.0f}%)\n')
                    f.write(f'    Flagged: {flagged_entries} ({flagged_entries/max(total_entries,1)*100:.0f}%)\n')
                    f.write(f'    Labels: {dict(labels)}\n')

    print(f'\n  DONE: {video_path.name}')
    print(f'  Annotated video: {out_video_path}')
    print(f'  Timeline CSV:    {timeline_path}')
    print(f'  Summary:         {summary_path}')
    print(f'  Speed: {fps_proc:.1f} fps ({elapsed:.1f} sec)')



# MAIN


def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Live Inference Pipeline for Exam Proctoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single video
    python phase4_inference.py --video exam_test1.mp4

    # Process with live preview
    python phase4_inference.py --video exam_test1.mp4 --show

    # Process all videos in a directory
    python phase4_inference.py --video_dir ./03_long_test_videos/

    # Use webcam
    python phase4_inference.py --webcam

    # Custom inference package path
    python phase4_inference.py --video exam.mp4 --pkg ./inference_package_v9.pth
        """
    )
    parser.add_argument('--video', type=str, help='Path to a single video file')
    parser.add_argument('--video_dir', type=str, help='Path to a directory of video files')
    parser.add_argument('--webcam', action='store_true', help='Use webcam as input')
    parser.add_argument('--pkg', type=str, default=DEFAULT_INFERENCE_PKG,
                        help=f'Path to inference package (default: {DEFAULT_INFERENCE_PKG})')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--show', action='store_true',
                        help='Show live preview window')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override Stage 1 threshold (default: use value from package)')

    args = parser.parse_args()

    # Validate inputs
    if not args.video and not args.video_dir and not args.webcam:
        parser.error('Provide --video, --video_dir, or --webcam')

    if not os.path.exists(args.pkg):
        print(f'ERROR: Inference package not found: {args.pkg}')
        print(f'Copy inference_package_v9.pth from SageMaker training_outputs/ to this directory.')
        sys.exit(1)

    # Load engine
    engine = InferenceEngine(args.pkg)
    if args.threshold is not None:
        engine.threshold = args.threshold
        print(f'  Threshold overridden to: {args.threshold:.2f}')

    # Process videos
    if args.video:
        if not os.path.exists(args.video):
            print(f'ERROR: Video not found: {args.video}')
            sys.exit(1)
        process_video(engine, args.video, args.output, show=args.show)

    elif args.video_dir:
        video_dir = Path(args.video_dir)
        videos = sorted(video_dir.glob('*.mp4'))
        if not videos:
            print(f'No .mp4 files found in {video_dir}')
            sys.exit(1)
        print(f'Found {len(videos)} videos in {video_dir}\n')
        for vp in videos:
            process_video(engine, vp, args.output, show=args.show)
            print()

    elif args.webcam:
        print('Webcam mode — press Q to quit')
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('ERROR: Could not open webcam')
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        engine.tracker = PersonTracker()
        engine.person_states = {}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = engine.process_frame(frame, fps)
            annotated = engine.draw_annotations(frame, results, frame_idx, fps)
            cv2.imshow('Phase 4 — Webcam', annotated)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
            frame_idx += 1
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Import pandas used for summary generation
    try:
        import pandas as pd
    except ImportError:
        pd = None
    main()
