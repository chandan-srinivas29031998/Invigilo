"""
test_crop.py
Opens a video, saves the original frame with a crop line and the
cropped result so you can check if the crop percentage is correct.
"""

import cv2
import sys
import os

CROP_TOP_PERCENT = 40

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_crop.py <video_file> [crop_percent]")
        print("Example: python test_crop.py 01_raw_scenario_videos/N01.mp4")
        print("Example: python test_crop.py 01_raw_scenario_videos/N01.mp4 50")
        sys.exit(1)

    video_path = sys.argv[1]
    crop_pct = int(sys.argv[2]) if len(sys.argv) > 2 else CROP_TOP_PERCENT

    if not os.path.exists(video_path):
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 5))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame from video")
        sys.exit(1)

    h, w = frame.shape[:2]
    crop_px = int(h * crop_pct / 100)
    cropped = frame[crop_px:, :]

    print(f"Original: {w}x{h}")
    print(f"Crop: top {crop_pct}% ({crop_px}px removed)")
    print(f"Result: {w}x{h - crop_px}")

    # Draw a red line on the original showing where the crop happens
    original_marked = frame.copy()
    cv2.line(original_marked, (0, crop_px), (w, crop_px), (0, 0, 255), 3)
    cv2.putText(original_marked, f"CROP LINE ({crop_pct}%)", (10, crop_px - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite("test_original_with_cropline.jpg", original_marked)
    cv2.imwrite("test_cropped_result.jpg", cropped)

    print("\nSaved:")
    print("  test_original_with_cropline.jpg  <- original with red crop line")
    print("  test_cropped_result.jpg          <- what the clips will look like")
    print(f"\nTo try a different crop: python test_crop.py {video_path} <percent>")


if __name__ == "__main__":
    main()
