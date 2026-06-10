"""

STEP 1: Pose Keypoint Extraction from Exam Clips

This script extracts body pose keypoints from every clip in this dataset.
For each clip it:
  1. Uses YOLOv8 to detect all people (bounding boxes)
  2. Crops each detected person
  3. Runs MediaPipe Pose to extract 33 keypoints per person per frame
  4. Saves structured .npz files for training the behaviour classification model.

"""

import os
import csv
import time
import argparse
import numpy as np
import cv2
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path


# --- Configuration ---

MAX_PEOPLE = 8          # Max people to track per frame (you have 8 seats)
NUM_KEYPOINTS = 33      # MediaPipe Pose landmarks
YOLO_CONF_THRESHOLD = 0.5  # Confidence threshold for person detection
PERSON_CLASS_ID = 0     # COCO class ID for "person"
SAMPLE_EVERY_N = 1      # Process every Nth frame (1 = all frames, 2 = every other)


def load_clip_metadata(csv_path):
    """Load clip_labels_master.csv and filter to usable clips only."""
    clips = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("review_status", "").strip()
            if status == "complete":
                clips.append(row)

    print(f"Loaded {len(clips)} usable clips (excluded clips filtered out)")
    return clips


def init_models():
    """Initialise YOLOv8 and MediaPipe Pose."""
    print("Loading YOLOv8 nano model...")
    yolo = YOLO("yolov8n.pt")

    print("Initialising MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,      # Video mode — uses temporal smoothing
        model_complexity=1,           # 0=lite, 1=full, 2=heavy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return yolo, pose


def detect_people(yolo, frame):
    """
    Run YOLOv8 on a frame to get person bounding boxes.
    Returns: list of (x1, y1, x2, y2, confidence) sorted left-to-right.
    """
    results = yolo(frame, verbose=False, classes=[PERSON_CLASS_ID], conf=YOLO_CONF_THRESHOLD)

    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))

    # Sort left-to-right by x1 - gives consistent ordering across frames
    boxes.sort(key=lambda b: b[0])

    return boxes[:MAX_PEOPLE]


def extract_pose(pose_model, frame, bbox):
    """
    Crop a person from the frame and extract MediaPipe Pose keypoints.
    
    Returns: (33, 4) array of [x, y, z, visibility] normalised to crop,
             or zeros if pose detection fails.
    """
    x1, y1, x2, y2, _ = bbox
    h, w = frame.shape[:2]

    # Clamp to frame the boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)

    # MediaPipe expects RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose_model.process(crop_rgb)

    keypoints = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)

    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks.landmark):
            # x, y are normalised to crop (0-1)
            # z is depth relative to hip midpoint
            # visibility is confidence that the landmark is visible
            keypoints[i] = [lm.x, lm.y, lm.z, lm.visibility]

    return keypoints


def process_clip(clip_path, yolo, pose_model):
    """
    Process a single video clip.
    
    Returns:
        keypoints:    (num_frames, MAX_PEOPLE, 33, 4)
        bboxes:       (num_frames, MAX_PEOPLE, 4)
        num_detected: (num_frames,)
        fps:          float
        total_frames: int
    """
    cap = cv2.VideoCapture(str(clip_path))

    if not cap.isOpened():
        print(f"    ERROR: Could not open {clip_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pre-allocate arrays
    sampled_frames = total_frames // SAMPLE_EVERY_N
    all_keypoints = np.zeros((sampled_frames, MAX_PEOPLE, NUM_KEYPOINTS, 4), dtype=np.float32)
    all_bboxes = np.zeros((sampled_frames, MAX_PEOPLE, 4), dtype=np.float32)
    all_num_detected = np.zeros(sampled_frames, dtype=np.int32)

    frame_idx = 0
    sampled_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if sampling
        if frame_idx % SAMPLE_EVERY_N != 0:
            frame_idx += 1
            continue

        if sampled_idx >= sampled_frames:
            break

        # Step 1: Detect all people
        boxes = detect_people(yolo, frame)
        all_num_detected[sampled_idx] = len(boxes)

        # Step 2: Extract pose for each detected person
        for person_idx, bbox in enumerate(boxes):
            kp = extract_pose(pose_model, frame, bbox)
            all_keypoints[sampled_idx, person_idx] = kp
            all_bboxes[sampled_idx, person_idx] = bbox[:4]

        frame_idx += 1
        sampled_idx += 1

    cap.release()

    # Trim to actual frames processed
    all_keypoints = all_keypoints[:sampled_idx]
    all_bboxes = all_bboxes[:sampled_idx]
    all_num_detected = all_num_detected[:sampled_idx]

    return all_keypoints, all_bboxes, all_num_detected, fps, total_frames


def main(dataset_root, csv_filename="clip_labels_master.csv"):
    """Main extraction loop."""
    dataset_root = Path(dataset_root)
    csv_path = dataset_root / csv_filename

    if not csv_path.exists():
        print(f"ERROR: Could not find {csv_path}")
        print(f"Make sure clip_labels_master.csv is in {dataset_root}")
        return

    # Create output directory
    poses_dir = dataset_root / "poses"
    poses_dir.mkdir(exist_ok=True)

    # Load metadata
    clips = load_clip_metadata(csv_path)

    # Init models
    yolo, pose_model = init_models()

    # Track results for summary
    summary_rows = []
    failed_clips = []

    print(f"\nProcessing {len(clips)} clips...")
    print(f"Output directory: {poses_dir}")
    print(f"Frame sampling: every {SAMPLE_EVERY_N} frame(s)")
    print("=" * 60)

    start_time = time.time()

    for i, clip_info in enumerate(clips):
        clip_name = clip_info["clip_name"]
        clip_rel_path = clip_info["clip_path"].replace("\\", os.sep)
        behavior = clip_info["behavior_class"]
        target_seat = clip_info.get("target_seat_id", "")
        hard_neg = clip_info.get("hard_negative_type", "")
        scenario_id = clip_info.get("scenario_id", "")

        clip_path = dataset_root / clip_rel_path

        # Progress
        elapsed = time.time() - start_time
        rate = (i / elapsed) if elapsed > 0 and i > 0 else 0
        eta = ((len(clips) - i) / rate / 60) if rate > 0 else 0
        print(f"  [{i+1}/{len(clips)}] {clip_name} ({behavior}) — ETA: {eta:.1f} min")

        if not clip_path.exists():
            print(f"    WARNING: File not found at {clip_path}")
            failed_clips.append(clip_name)
            continue

        # Process
        result = process_clip(clip_path, yolo, pose_model)

        if result is None:
            failed_clips.append(clip_name)
            continue

        keypoints, bboxes, num_detected, fps, total_frames = result

        # Save as .npz
        out_path = poses_dir / f"{Path(clip_name).stem}.npz"
        np.savez_compressed(
            out_path,
            keypoints=keypoints,
            bbox=bboxes,
            num_detected=num_detected,
            fps=np.array(fps),
            total_frames=np.array(total_frames),
            behavior_class=np.array(behavior),
            target_seat=np.array(target_seat),
            scenario_id=np.array(scenario_id),
            hard_negative_type=np.array(hard_neg),
            clip_name=np.array(clip_name),
        )

        # Summary stats
        avg_people = np.mean(num_detected)
        min_people = np.min(num_detected)
        max_people = np.max(num_detected)

        summary_rows.append({
            "clip_name": clip_name,
            "behavior_class": behavior,
            "target_seat": target_seat,
            "scenario_id": scenario_id,
            "hard_negative_type": hard_neg,
            "total_frames": total_frames,
            "sampled_frames": len(keypoints),
            "fps": round(fps, 1),
            "avg_people_detected": round(avg_people, 1),
            "min_people_detected": int(min_people),
            "max_people_detected": int(max_people),
            "npz_path": str(out_path.relative_to(dataset_root)),
        })

    # Write summary CSV
    summary_path = poses_dir / "extraction_summary.csv"
    fieldnames = [
        "clip_name", "behavior_class", "target_seat", "scenario_id",
        "hard_negative_type", "total_frames", "sampled_frames", "fps",
        "avg_people_detected", "min_people_detected", "max_people_detected",
        "npz_path",
    ]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Final report
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Clips processed:  {len(summary_rows)}")
    print(f"  Clips failed:     {len(failed_clips)}")
    print(f"  Total time:       {total_time/60:.1f} minutes")
    print(f"  Output directory:  {poses_dir}")
    print(f"  Summary CSV:       {summary_path}")

    if failed_clips:
        print(f"\n  Failed clips:")
        for fc in failed_clips:
            print(f"    - {fc}")

    # Print class distribution
    from collections import Counter
    class_counts = Counter(r["behavior_class"] for r in summary_rows)
    print(f"\n  Class distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {count}")

    # Print detection quality
    avg_detected = np.mean([r["avg_people_detected"] for r in summary_rows])
    low_detect = [r for r in summary_rows if r["min_people_detected"] < 6]
    print(f"\n  Average people detected per frame: {avg_detected:.1f}")
    print(f"  Clips with <6 people in some frames: {len(low_detect)}")

    print(f"\n  Next step: Upload extraction_summary.csv to Claude")
    print(f"  and I will build the training notebook for you.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pose keypoints from exam behaviour clips."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./exam_proctoring_dataset",
        help="Root directory of the dataset (contains clip_labels_master.csv and 02_clips/)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="clip_labels_master.csv",
        help="Name of the clip labels CSV file",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=1,
        help="Process every Nth frame (1=all, 2=half, 3=every third). Use 2 or 3 to speed up.",
    )

    args = parser.parse_args()
    SAMPLE_EVERY_N = args.sample_every
    main(args.dataset_root, args.csv)
