"""

Pose Quality Checker

Run this AFTER extract_poses.py finishes.

Checks the visibility scores for upper-body keypoints that matter for behaviour classification. Outputs a CSV with per-clip quality scores so you can decide which clips to exclude.

"""

import csv
import argparse
import numpy as np
from pathlib import Path

# The keypoints that matter for your 7 behaviour classes
# Index : Name              : Why it matters
# 0     : nose              : head direction (looking_sideways, looking_down)
# 2     : left_eye          : gaze proxy
# 5     : right_eye         : gaze proxy
# 7     : left_ear          : head rotation indicator
# 8     : right_ear         : head rotation indicator
# 9     : mouth_left        : talking detection
# 10    : mouth_right       : talking detection
# 11    : left_shoulder     : torso reference, leaning
# 12    : right_shoulder    : torso reference, leaning
# 13    : left_elbow        : arm position
# 14    : right_elbow       : arm position
# 15    : left_wrist        : phone/note detection
# 16    : right_wrist       : phone/note detection
# 23    : left_hip          : torso angle for leaning
# 24    : right_hip         : torso angle for leaning

CRITICAL_KEYPOINTS = {
    0:  "nose",
    2:  "left_eye",
    5:  "right_eye",
    7:  "left_ear",
    8:  "right_ear",
    9:  "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}

QUALITY_THRESHOLD = 0.5   # Below this threshold = low quality


def analyze_clip(npz_path):
    """Analyze pose quality for a single clip."""
    data = np.load(npz_path, allow_pickle=True)

    kp = data["keypoints"]              # (frames, max_people, 33, 4)
    num_detected = data["num_detected"] # (frames,)
    behavior = str(data["behavior_class"])
    target_seat = str(data["target_seat"])
    scenario_id = str(data["scenario_id"])

    num_frames = kp.shape[0]

    # Visibility scores for critical keypoints across ALL detected people
    # Shape: (frames, max_people, num_critical)
    critical_indices = list(CRITICAL_KEYPOINTS.keys())
    vis_all = kp[:, :, critical_indices, 3]

    # Overall average visibility (across all people, all frames, all critical kps)
    overall_avg = float(np.mean(vis_all[vis_all > 0])) if np.any(vis_all > 0) else 0.0

    # Per-keypoint average visibility (across all people and frames)
    per_kp_avg = {}
    for i, kp_idx in enumerate(critical_indices):
        kp_name = CRITICAL_KEYPOINTS[kp_idx]
        vals = vis_all[:, :, i]
        avg = float(np.mean(vals[vals > 0])) if np.any(vals > 0) else 0.0
        per_kp_avg[kp_name] = round(avg, 3)

    # Count frames where nose visibility is very low (head might be clipped)
    nose_vis = kp[:, :, 0, 3]  # (frames, max_people)
    # For each frame, check if any detected person has nose vis < 0.3
    frames_with_bad_nose = 0
    for f in range(num_frames):
        n_det = int(num_detected[f])
        if n_det > 0:
            frame_nose_vis = nose_vis[f, :n_det]
            if np.any(frame_nose_vis < 0.3):
                frames_with_bad_nose += 1

    bad_nose_pct = round(frames_with_bad_nose / max(num_frames, 1) * 100, 1)

    # Count frames where wrist visibility is low (arms clipped)
    wrist_vis = kp[:, :, [15, 16], 3]  # (frames, max_people, 2)
    frames_with_bad_wrists = 0
    for f in range(num_frames):
        n_det = int(num_detected[f])
        if n_det > 0:
            frame_wrist_vis = wrist_vis[f, :n_det, :]
            if np.any(frame_wrist_vis < 0.3):
                frames_with_bad_wrists += 1

    bad_wrist_pct = round(frames_with_bad_wrists / max(num_frames, 1) * 100, 1)

    # Quality verdict
    if overall_avg >= 0.7:
        quality = "good"
    elif overall_avg >= QUALITY_THRESHOLD:
        quality = "ok"
    else:
        quality = "poor"

    return {
        "clip_name": npz_path.stem,
        "behavior_class": behavior,
        "target_seat": target_seat,
        "scenario_id": scenario_id,
        "num_frames": num_frames,
        "overall_visibility": round(overall_avg, 3),
        "quality": quality,
        "bad_nose_frames_pct": bad_nose_pct,
        "bad_wrist_frames_pct": bad_wrist_pct,
        **per_kp_avg,
    }


def main(poses_dir):
    poses_path = Path(poses_dir)
    npz_files = sorted(poses_path.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {poses_dir}")
        return

    print(f"Analyzing {len(npz_files)} clips...")

    rows = []
    for i, npz_path in enumerate(npz_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(npz_files)}] {npz_path.stem}")
        row = analyze_clip(npz_path)
        rows.append(row)

    # Write CSV
    output_path = poses_path / "pose_quality_report.csv"
    fieldnames = [
        "clip_name", "behavior_class", "target_seat", "scenario_id",
        "num_frames", "overall_visibility", "quality",
        "bad_nose_frames_pct", "bad_wrist_frames_pct",
    ] + list(CRITICAL_KEYPOINTS.values())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    good = len([r for r in rows if r["quality"] == "good"])
    ok = len([r for r in rows if r["quality"] == "ok"])
    poor = len([r for r in rows if r["quality"] == "poor"])

    print(f"\n{'='*60}")
    print("POSE QUALITY REPORT")
    print(f"{'='*60}")
    print(f"  Total clips:    {len(rows)}")
    print(f"  Good (>0.7):    {good}")
    print(f"  OK (0.5-0.7):   {ok}")
    print(f"  Poor (<0.5):    {poor}")

    # Per-class breakdown
    print(f"\n  Per-class quality:")
    classes = sorted(set(r["behavior_class"] for r in rows))
    for cls in classes:
        cls_rows = [r for r in rows if r["behavior_class"] == cls]
        avg = np.mean([r["overall_visibility"] for r in cls_rows])
        poor_count = len([r for r in cls_rows if r["quality"] == "poor"])
        print(f"    {cls}: avg visibility {avg:.3f}, {poor_count} poor clips")

    # Per-seat breakdown
    print(f"\n  Per-seat quality:")
    seats = sorted(set(r["target_seat"] for r in rows if r["target_seat"]))
    for seat in seats:
        seat_rows = [r for r in rows if r["target_seat"] == seat]
        avg = np.mean([r["overall_visibility"] for r in seat_rows])
        print(f"    {seat}: avg visibility {avg:.3f} ({len(seat_rows)} clips)")

    # Worst clips
    worst = sorted(rows, key=lambda r: r["overall_visibility"])[:10]
    print(f"\n  10 worst clips:")
    for r in worst:
        print(f"    {r['clip_name']}: visibility {r['overall_visibility']}, "
              f"nose bad {r['bad_nose_frames_pct']}%, wrist bad {r['bad_wrist_frames_pct']}%")

    print(f"\n  Report saved to: {output_path}")
    print(f"  Recommendation: exclude clips with quality='poor' from training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pose extraction quality")
    parser.add_argument(
        "--poses_dir", type=str, default="./poses",
        help="Path to poses directory with .npz files",
    )
    args = parser.parse_args()
    main(args.poses_dir)
