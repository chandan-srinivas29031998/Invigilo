"""
split_clips.py
Splits raw scenario videos into 10-second clips, crops the top portion
of the frame (empty seats/ceiling), and fills clip_labels_master.csv.

No resizing is done — the model's dataloader handles that during training.
This avoids stretching, black bars, and wasted pixels.

PER-VIDEO CROP OVERRIDE:
------------------------
If different videos need different crop amounts, create a file called
crop_overrides.csv in the same folder:

    scenario_id,crop_percent
    N01,40
    P03,55
    LD01,50
"""

import os
import csv
import cv2
import sys

# CONFIGURATION

ROOT_DIR = "."  # Change this if your dataset folder is elsewhere
RAW_VIDEO_DIR = os.path.join(ROOT_DIR, "01_raw_scenario_videos")
CLIPS_DIR = os.path.join(ROOT_DIR, "02_clips")
SHOT_LIST_PATH = os.path.join(ROOT_DIR, "shot_list.csv")
MASTER_CSV_PATH = os.path.join(ROOT_DIR, "clip_labels_master.csv")
CROP_OVERRIDES_PATH = os.path.join(ROOT_DIR, "crop_overrides.csv")

CLIP_DURATION = 10       # seconds per clip
CROP_TOP_PERCENT = 50    


#LOAD HELPERS

def load_shot_list(path):
    """Read shot_list.csv and return a list of scenario dicts."""
    scenarios = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            scenarios.append(cleaned)
    return scenarios


def load_crop_overrides(path):
    """Read crop_overrides.csv and return dict of scenario_id -> crop_percent."""
    overrides = {}
    if not os.path.exists(path):
        return overrides
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["scenario_id"].strip().upper()
            pct = int(row["crop_percent"].strip())
            overrides[sid] = pct
    print(f"Loaded {len(overrides)} crop overrides from {path}")
    return overrides


# SPLIT A SINGLE VIDEO 
def split_video(video_path, scenario_id, behavior_class, hard_negative_type,
                num_takes, output_dir, crop_percent):
    """
    Split a single video into num_takes clips of CLIP_DURATION seconds each.
    Each clip has the top portion cropped. No resizing is applied.
    Returns a list of dicts (one per clip) for the master CSV.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_clip = int(fps * CLIP_DURATION)

    # Calculate crop
    crop_top_px = int(orig_height * crop_percent / 100)
    cropped_height = orig_height - crop_top_px
    output_size = (orig_width, cropped_height)

    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  FPS: {fps:.1f} | Duration: {total_duration:.1f}s | Original: {orig_width}x{orig_height}")
    print(f"  Crop: top {crop_percent}% ({crop_top_px}px) -> Output: {orig_width}x{cropped_height}")
    print(f"  Expected: {num_takes} clips x {CLIP_DURATION}s = {num_takes * CLIP_DURATION}s")

    # Check if video is long enough
    required_duration = num_takes * CLIP_DURATION
    if total_duration < required_duration - 2:
        print(f"  WARNING: Video is {total_duration:.1f}s but need ~{required_duration}s for {num_takes} clips")

    # Create output subfolder
    class_dir = os.path.join(output_dir, behavior_class)
    os.makedirs(class_dir, exist_ok=True)

    clips_info = []

    for take_num in range(1, num_takes + 1):
        start_frame = (take_num - 1) * frames_per_clip
        end_frame = start_frame + frames_per_clip

        # Build clip filename
        take_str = f"take{take_num:02d}"
        clip_name = f"{scenario_id}_{behavior_class}_{take_str}.mp4"
        clip_path = os.path.join(class_dir, clip_name)
        relative_path = os.path.join("02_clips", behavior_class, clip_name)

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Set up writer at cropped resolution (no resize)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(clip_path, fourcc, fps, output_size)

        frames_written = 0
        while frames_written < frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break
            # Crop the top portion only
            if crop_top_px > 0:
                frame = frame[crop_top_px:, :]
            writer.write(frame)
            frames_written += 1

        writer.release()

        actual_duration = frames_written / fps

        # Skip if clip is too short (less than 8 seconds)
        if actual_duration < 8:
            print(f"  WARNING: {clip_name} is only {actual_duration:.1f}s — skipping (too short)")
            if os.path.exists(clip_path):
                os.remove(clip_path)
            continue

        print(f"  Created: {clip_name} ({frames_written} frames, {actual_duration:.1f}s)")

        # Build CSV row
        clip_info = {
            "clip_name": clip_name,
            "clip_path": relative_path,
            "behavior_class": behavior_class,
            "hard_negative_type": hard_negative_type if hard_negative_type else "",
            "scenario_id": scenario_id,
            "take_no": take_num,
            "target_seat_id": "",
            "neighbor_seat_id": "",
            "notes": "",
            "split": "",
            "review_status": "pending"
        }
        clips_info.append(clip_info)

    cap.release()
    return clips_info


# WRITE MASTER CSV

def write_master_csv(all_clips, output_path):
    """Write clip_labels_master.csv with all clip entries."""
    fieldnames = [
        "clip_name", "clip_path", "behavior_class", "hard_negative_type",
        "scenario_id", "take_no", "target_seat_id", "neighbor_seat_id",
        "notes", "split", "review_status"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_clips)

    print(f"\nMaster CSV written: {output_path} ({len(all_clips)} clips)")


# MAIN
def main():
    print("=" * 60)
    print("CLIP SPLITTER")
    print(f"Default crop: top {CROP_TOP_PERCENT}% | No resize")
    print("=" * 60)

    # Validate paths
    if not os.path.exists(RAW_VIDEO_DIR):
        print(f"\nERROR: Raw video folder not found: {RAW_VIDEO_DIR}")
        print("Create it and place your scenario videos inside.")
        sys.exit(1)

    if not os.path.exists(SHOT_LIST_PATH):
        print(f"\nERROR: Shot list not found: {SHOT_LIST_PATH}")
        sys.exit(1)

    # Load shot list and crop overrides
    scenarios = load_shot_list(SHOT_LIST_PATH)
    print(f"\nLoaded {len(scenarios)} scenarios from shot_list.csv")

    crop_overrides = load_crop_overrides(CROP_OVERRIDES_PATH)

    # List available videos
    available_videos = {}
    for f in os.listdir(RAW_VIDEO_DIR):
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            name_without_ext = os.path.splitext(f)[0]
            available_videos[name_without_ext.upper()] = f
    print(f"Found {len(available_videos)} raw videos in {RAW_VIDEO_DIR}\n")

    # Ask for confirmation
    print(f"Ready to process {len(scenarios)} scenarios.")
    response = input("Continue? (y/n): ").strip().lower()
    if response != "y":
        print("Aborted.")
        sys.exit(0)

    # Process each scenario
    os.makedirs(CLIPS_DIR, exist_ok=True)
    all_clips = []
    processed = 0
    skipped = 0

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"].upper()
        behavior_class = scenario["behavior_class"]
        hard_negative_type = scenario.get("hard_negative_type", "")
        num_takes = int(scenario["required_takes"])

        if scenario_id not in available_videos:
            print(f"\n[SKIP] {scenario_id}: No video file found")
            skipped += 1
            continue

        # Use override crop if available, otherwise default
        crop_pct = crop_overrides.get(scenario_id, CROP_TOP_PERCENT)

        video_filename = available_videos[scenario_id]
        video_path = os.path.join(RAW_VIDEO_DIR, video_filename)

        print(f"\n[{scenario_id}] {behavior_class} — {num_takes} takes (crop: {crop_pct}%)")

        clips = split_video(
            video_path=video_path,
            scenario_id=scenario_id,
            behavior_class=behavior_class,
            hard_negative_type=hard_negative_type,
            num_takes=num_takes,
            output_dir=CLIPS_DIR,
            crop_percent=crop_pct
        )

        all_clips.extend(clips)
        processed += 1

    # Write master CSV
    if all_clips:
        write_master_csv(all_clips, MASTER_CSV_PATH)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Scenarios processed: {processed}")
    print(f"Scenarios skipped (no video found): {skipped}")
    print(f"Total clips created: {len(all_clips)}")
    print(f"Clips saved to: {CLIPS_DIR}")
    print(f"Master CSV: {MASTER_CSV_PATH}")

    # Per class summary
    class_counts = {}
    for clip in all_clips:
        cls = clip["behavior_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print("\nClips per class:")
    for cls in sorted(class_counts.keys()):
        print(f"  {cls}: {class_counts[cls]}")


if __name__ == "__main__":
    main()
