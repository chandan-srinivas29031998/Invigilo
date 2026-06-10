"""
Pose Extraction Visualiser
Overlays extracted keypoints and bounding boxes on video frames so you can
visually verify that extraction worked correctly.

Controls:
    D / Right Arrow  → Next frame
    A / Left Arrow   → Previous frame
    Space            → Play / Pause auto-advance
    Q / Esc          → Quit
    S                → Save current frame as PNG
    + / =            → Speed up playback
    - / _            → Slow down playback
"""

import argparse
import sys
import numpy as np
import cv2
from pathlib import Path


# MediaPipe Pose Connections (upper body focus) 

# All 33 MediaPipe connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),        # left face
    (0, 4), (4, 5), (5, 6), (6, 8),        # right face
    (9, 10),                                 # mouth
    (11, 12),                                # shoulders
    (11, 13), (13, 15),                      # left arm
    (12, 14), (14, 16),                      # right arm
    (11, 23), (12, 24),                      # torso
    (23, 24),                                # hips
    (15, 17), (15, 19), (15, 21),           # left hand
    (16, 18), (16, 20), (16, 22),           # right hand
    (23, 25), (25, 27), (27, 29), (27, 31), # left leg
    (24, 26), (26, 28), (28, 30), (28, 32), # right leg
]

# Upper body connections (the ones that matter)
UPPER_BODY = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
]

# Keypoint names for display
KP_NAMES = {
    0: "nose", 2: "L eye", 5: "R eye",
    7: "L ear", 8: "R ear",
    9: "mouth L", 10: "mouth R",
    11: "L shoulder", 12: "R shoulder",
    13: "L elbow", 14: "R elbow",
    15: "L wrist", 16: "R wrist",
    23: "L hip", 24: "R hip",
}

# Colors for each person (BGR)
PERSON_COLORS = [
    (0, 255, 0),     # green
    (255, 165, 0),   # orange
    (0, 255, 255),   # yellow
    (255, 0, 0),     # blue
    (255, 0, 255),   # magenta
    (0, 165, 255),   # gold
    (128, 0, 255),   # purple
    (255, 255, 0),   # cyan
]


def find_clip_files(dataset_root, clip_name):
    """Find the video and npz file for a given clip name."""
    dataset_root = Path(dataset_root)

    # Strip extension if provided
    clip_stem = Path(clip_name).stem

    # Find the npz file
    npz_path = dataset_root / "poses" / f"{clip_stem}.npz"
    if not npz_path.exists():
        print(f"ERROR: Could not find {npz_path}")
        sys.exit(1)

    # Find the video file by searching 02_clips subdirectories
    video_path = None
    clips_dir = dataset_root / "02_clips"
    if clips_dir.exists():
        for mp4 in clips_dir.rglob(f"{clip_stem}.mp4"):
            video_path = mp4
            break

    if video_path is None:
        print(f"ERROR: Could not find video {clip_stem}.mp4 in {clips_dir}")
        sys.exit(1)

    print(f"Video: {video_path}")
    print(f"Poses: {npz_path}")

    return video_path, npz_path


def draw_frame(frame, keypoints, bboxes, num_detected, frame_idx, total_frames,
               behavior, target_seat, show_labels=False, selected_person=-1):
    """
    Draw bounding boxes, skeleton, and keypoints on a frame.

    Args:
        frame:           BGR image
        keypoints:       (max_people, 33, 4) for this frame
        bboxes:          (max_people, 4) for this frame
        num_detected:    int — number of people detected
        frame_idx:       current frame number
        total_frames:    total frames in clip
        behavior:        behavior class string
        target_seat:     target seat string
        show_labels:     whether to show keypoint names
        selected_person: show labels/detail for this person only (-1 = all)
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for person_idx in range(int(num_detected)):
        color = PERSON_COLORS[person_idx % len(PERSON_COLORS)]
        kp = keypoints[person_idx]   # (33, 4)
        bbox = bboxes[person_idx]    # (4,)

        # Is this the person we're focusing on?
        is_selected = (selected_person == -1 or selected_person == person_idx)

        # Draw bounding box
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 > x1 and y2 > y1:
            box_thickness = 2 if is_selected else 1
            box_color = color if is_selected else tuple(int(c * 0.4) for c in color)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, box_thickness)

            # Person label
            label = f"P{person_idx + 1}"
            cv2.putText(overlay, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Compute pixel coordinates from normalised keypoints
        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w <= 0 or crop_h <= 0:
            continue

        kp_pixels = []
        for ki in range(33):
            kx, ky, kz, vis = kp[ki]
            if vis > 0.3:
                px = int(x1 + kx * crop_w)
                py = int(y1 + ky * crop_h)
                kp_pixels.append((px, py, vis))
            else:
                kp_pixels.append(None)

        # Draw skeleton connections
        for (a, b) in UPPER_BODY:
            if kp_pixels[a] is not None and kp_pixels[b] is not None:
                pa = kp_pixels[a]
                pb = kp_pixels[b]
                if is_selected:
                    thickness = 2 if min(pa[2], pb[2]) > 0.7 else 1
                    line_color = tuple(int(c * 0.7) for c in color)
                else:
                    thickness = 1
                    line_color = tuple(int(c * 0.3) for c in color)
                cv2.line(overlay, (pa[0], pa[1]), (pb[0], pb[1]),
                         line_color, thickness)

        # Draw keypoints
        for ki, pt in enumerate(kp_pixels):
            if pt is None:
                continue
            px, py, vis = pt

            if is_selected:
                radius = 4 if vis > 0.7 else 2
                cv2.circle(overlay, (px, py), radius, color, -1)
                cv2.circle(overlay, (px, py), radius, (0, 0, 0), 1)
            else:
                cv2.circle(overlay, (px, py), 2, tuple(int(c * 0.4) for c in color), -1)

            # Show keypoint labels only for selected person when toggled on
            if show_labels and is_selected and ki in KP_NAMES and vis > 0.5:
                text = KP_NAMES[ki]
                # Dark outline for readability, then colored text on top
                cv2.putText(overlay, text, (px + 7, py - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                cv2.putText(overlay, text, (px + 7, py - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Info bar at the top
    bar_h = 40
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)

    info_text = (f"Frame {frame_idx + 1}/{total_frames}  |  "
                 f"Detected: {int(num_detected)}  |  "
                 f"Class: {behavior}  |  "
                 f"Seat: {target_seat}")
    if selected_person >= 0:
        info_text += f"  |  Selected: P{selected_person + 1}"
    if show_labels:
        info_text += "  |  Labels: ON"
    cv2.putText(overlay, info_text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # Controls hint at the bottom
    hint_h = 25
    cv2.rectangle(overlay, (0, h - hint_h), (w, h), (30, 30, 30), -1)
    cv2.putText(overlay, "A/D: prev/next  |  Space: play  |  1-8: select person  |  L: labels  |  0: show all  |  Q: quit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    return overlay


def visualise(dataset_root, clip_name, start_frame=0):
    """Main visualisation loop."""
    video_path, npz_path = find_clip_files(dataset_root, clip_name)

    # Load extracted data
    data = np.load(npz_path, allow_pickle=True)
    keypoints = data["keypoints"]        # (frames, 8, 33, 4)
    bboxes = data["bbox"]                # (frames, 8, 4)
    num_detected = data["num_detected"]  # (frames,)
    behavior = str(data["behavior_class"])
    target_seat = str(data["target_seat"])

    total_pose_frames = len(keypoints)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nClip: {clip_name}")
    print(f"Behaviour: {behavior}")
    print(f"Target seat: {target_seat}")
    print(f"Video frames: {total_video_frames}")
    print(f"Pose frames: {total_pose_frames}")
    print(f"FPS: {fps}")
    print(f"\nControls: A/D = prev/next, Space = play/pause, S = save, Q = quit\n")

    sample_ratio = max(1, total_video_frames // total_pose_frames)
    if sample_ratio > 1:
        print(f"Note: Poses were sampled every {sample_ratio} frames")

    frame_idx = start_frame
    playing = False
    delay = int(1000 / fps)  # ms per frame at original speed
    show_labels = False       # Labels OFF by default — press L to toggle
    selected_person = -1      # -1 = show all, 0-7 = focus on one person

    window_name = f"Pose Visualiser — {clip_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        # Seek to current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Could not read frame {frame_idx}")
            frame_idx = 0
            continue

        # Map video frame to pose frame
        pose_idx = frame_idx // sample_ratio
        pose_idx = min(pose_idx, total_pose_frames - 1)

        # Draw overlays
        display = draw_frame(
            frame,
            keypoints[pose_idx],
            bboxes[pose_idx],
            num_detected[pose_idx],
            frame_idx,
            total_video_frames,
            behavior,
            target_seat,
            show_labels=show_labels,
            selected_person=selected_person,
        )

        cv2.imshow(window_name, display)

        # Handle input
        wait_time = delay if playing else 0
        key = cv2.waitKey(max(1, wait_time)) & 0xFF

        if key == ord('q') or key == 27:  # Q or Esc
            break
        elif key == ord('d') or key == 83:  # D or Right arrow
            frame_idx = min(frame_idx + 1, total_video_frames - 1)
        elif key == ord('a') or key == 81:  # A or Left arrow
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord(' '):  # Space — toggle play
            playing = not playing
            print(f"{'Playing' if playing else 'Paused'} at frame {frame_idx}")
        elif key == ord('s'):  # Save frame
            save_path = f"{Path(clip_name).stem}_frame{frame_idx:04d}.png"
            cv2.imwrite(save_path, display)
            print(f"Saved: {save_path}")
        elif key == ord('+') or key == ord('='):  # Speed up
            delay = max(10, delay - 10)
            print(f"Delay: {delay}ms")
        elif key == ord('-') or key == ord('_'):  # Slow down
            delay = min(500, delay + 10)
            print(f"Delay: {delay}ms")
        elif key == ord('l'):  # Toggle labels
            show_labels = not show_labels
            print(f"Labels: {'ON' if show_labels else 'OFF'}")
        elif key == ord('0'):  # Show all people
            selected_person = -1
            print("Showing all people")
        elif key in [ord(str(i)) for i in range(1, 9)]:  # 1-8: select person
            selected_person = key - ord('1')  # Convert '1' -> 0, '2' -> 1, etc.
            print(f"Selected: P{selected_person + 1}")
        elif playing:
            frame_idx = min(frame_idx + 1, total_video_frames - 1)
            if frame_idx >= total_video_frames - 1:
                playing = False
                print("Reached end of clip")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise extracted pose keypoints on video frames."
    )
    parser.add_argument(
        "clip",
        type=str,
        help="Clip name (e.g., L03_looking_sideways_take03)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=".",
        help="Root directory of dataset (default: current directory)",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Frame to start at (default: 0)",
    )

    args = parser.parse_args()
    visualise(args.dataset_root, args.clip, args.start_frame)
