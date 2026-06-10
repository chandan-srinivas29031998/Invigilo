# Invigilo — AI Exam Proctoring System

> AI-assisted suspicious behaviour detection in physical exam rooms using computer vision.
> Built as a university project for Deep Learning & CNN (42028) at UTS Sydney (Autumn 2026).

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Training-red)]()

**98.1% event detection rate** on 9 unseen test videos (107 ground-truth suspicious events).

---

## Problem

Traditional exam invigilation relies on human supervisors who cannot monitor every student simultaneously. A single invigilator watching dozens of students will miss brief events — a sideways glance, a whispered exchange, a note passed under a desk — especially over a multi-hour session.

Invigilo uses a pose-based computer vision pipeline to flag suspicious behaviour in real time, giving invigilators a second pair of eyes that never blinks. The system augments human judgment, it does not replace it.

---

## Results

| Metric | Value | Notes |
|---|---|---|
| Event detection rate | 98.1% (105/107) | At least one suspicious prediction during the event window |
| Sustained flagging rate | 68.2% (73/107) | Event triggers the 4-of-7 rule |
| Missed events | 1.9% (2/107) | Both 3-second events in TV08 |
| Hard-negative false flag | 69.4% (25/36) | Model flags during annotated normal fidgeting |
| Total test observations | 4,859 | One per person per second across 9 videos |

The system detects that suspicious behaviour is occurring in 98.1% of events. The gap between detection (98.1%) and flagging (68.2%) reflects the 4-of-7 sustained behaviour rule working as intended — filtering out brief or intermittent detections before raising an alert.

---

## System Architecture — 5-Stage Pipeline

| Stage | What it does | Model / Method |
|---|---|---|
| 1 — Person Detection | Detects all people in the frame, discards detections below 0.4 confidence | YOLOv8-nano |
| 2 — Tracking | Assigns stable IDs across frames; holds tracks open for ~7 sec during occlusion; graveyard-merge heuristic preserves classification history on re-entry | ByteTrack |
| 3 — Pose Estimation | Crops each tracked person and extracts 33 body keypoints (x, y, z + visibility); converts crop-normalised coordinates back to frame-pixel space | MediaPipe Pose |
| 4 — Feature Engineering | Computes 13 geometric features per frame (head turn angle, head turn velocity, torso lean, shoulder tilt, wrist drop, wrist lateral extent, ear visibility asymmetry, mouth movement, body movement); normalises by shoulder width; aggregates over 3-sec sliding windows into 78-dimensional vectors | Hand-crafted features |
| 5 — Classification & Flagging | Two-stage MLP: Stage 1 binary (normal vs suspicious, threshold 0.20 for high recall); Stage 2 subtype (head turn vs lateral movement); 4-of-7 sustained flagging rule before raising alert | Two-stage MLP (v11) |

---

## Model Experiments

Six architectures were trained and evaluated. v11 was selected for deployment.

| Version | Architecture | Classes | Test Acc | Macro F1 | Notes |
|---|---|---|---|---|---|
| v8 | Two-stage MLP | 5 suspicious | 75.7% | 0.474 | Baseline |
| v9 | Two-stage MLP | 3 merged | 78.8% | 0.634 | Merging improves recall |
| v11 | Two-stage MLP | 2 sus (no LD) | 82.4% | 0.691 | **Deployed** |
| v12 | 1D-CNN temporal | 4 classes | 79.8% | 0.683 | Temporal helps |
| v13 | 1D-CNN temporal | 3 classes | 83.1% | 0.690 | Highest accuracy |
| v14 | Bi-GRU temporal | 4 classes | 79.3% | 0.722 | Highest F1 |

v11 selected over v14 (higher macro F1) because the two-stage MLP is faster at inference, simpler to deploy, and removing the `looking_down` class cut false alarms substantially on long test videos.

---

## Dataset

No public dataset exists for exam-hall cheating detection with pose annotations. The team self-recorded 393 ten-second clips in a university lecture theatre with 8 volunteers across multiple scenarios.

| Category | Clips | Notes |
|---|---|---|
| Looking sideways | 62 | Merged into `head_turn` |
| Talking to neighbor | 38 | Merged into `head_turn` |
| Leaning to neighbor | 52 | Merged into `lateral_movement` |
| Passing note | 42 | Merged into `lateral_movement` |
| Looking down | 41 | Dropped in v11 (too many false alarms) |
| Using phone | 54 | Dropped from pose model (wrist signal identical to writing) |
| Normal + hard negatives | 104 | Scratching, stretching, page turning |
| **Total (after QA)** | **339** | |

9 separate test videos (1–2 min each, 5–7 students per video) were held out entirely during training.

---

## Tech Stack

- **Detection:** YOLOv8-nano (Ultralytics)
- **Tracking:** ByteTrack
- **Pose estimation:** MediaPipe Pose (33 keypoints)
- **Feature engineering:** Hand-crafted geometric features, 3-sec sliding windows
- **Classification:** Two-stage MLP, trained with PyTorch on AWS SageMaker
- **Web app:** Flask + MJPEG streaming (real-time annotated video)
- **Development:** Python 3.11, Jupyter Notebooks

---

## Repository Structure

```
Invigilo/
├── phase1_dataExtraction_processing_scripts/   Data collection and preprocessing
├── phase2_feature_extraction_notebooks/        Feature extraction — 13 geometric features, 78-dim vectors
├── phase3_training_notebooks/                  Model training experiments (v8–v14)
├── phase4_inference/                           Real-time inference pipeline (v1–v3)
├── phase5_GUI/                                 Flask web application
├── model/                                      Trained model weights
├── data/05_annotations_csv/                    Ground truth annotation files
└── docs/                                       Design documents and project reports
```

---

## My Contributions — Chandan Sreenivasaiah

My responsibilities spanned data management, ground truth annotation, quality assurance, model training, and evaluation.

**Dataset & Annotation**
- Created and maintained `clip_labels_master.csv` — the single source of truth for all 393 clips, tracking filename, behaviour class, target seat, scenario ID, and usability status across the full recording lifecycle
- Identified and resolved a shortcut learning problem where resolution differences across scenario groups caused the model to classify by aspect ratio distortion rather than behaviour; verified camera settings per scenario and corrected metadata so resolution normalisation could be applied correctly
- Created `long_video_event_log.csv` — frame-by-frame ground truth annotations for all 9 test videos (107 suspicious events + 36 hard-negative events), with precise start/end times, seat IDs, and behaviour classes; these labels are the benchmark against which all Phase 4 evaluation numbers are measured
- Created `manual_active_segments.csv` marking the active 4–6 second behaviour windows within each training clip, used by Phase 2 to focus feature extraction on relevant windows rather than treating all 10 seconds equally

**Quality Assurance**
- Managed `target_person_review.csv`; reviewed hundreds of pose visualisation images to verify correct target person identification per clip
- Identified and excluded 130 training windows where the target person was misidentified, occluded, or the behaviour was ambiguous — directly improving classifier signal quality

**Model Training**
- Trained the **v8 baseline experiment** — first working two-stage MLP with all 5 original suspicious classes (75.7% test accuracy, Macro F1 0.474); v8 results revealed 3 of 5 classes had recall below 27%, directly motivating the class-merging strategy used in v9 onwards
- Trained the **v14 Bidirectional GRU model** — 2-layer BiGRU with 64 hidden units per direction (128 total), 113K parameters; achieved the highest macro F1 of any experiment (0.722) and 97% recall on `looking_down`, demonstrating that temporal dependencies in both directions are key to distinguishing genuine suspicious glances from normal writing posture

**Evaluation**
- Wrote `evaluate_inference.py` — the evaluation script that matches model predictions against ground-truth event logs and computes detection rates, flagging rates, class accuracy, false alarm rates, detection latency, per-person breakdowns, temporal heatmaps, and summary charts
- Ran all evaluations across all three pipeline versions (v1, v2, v3) on the 9 test videos and analysed results to guide each iteration of pipeline improvements

---

## Team

| Member | Contribution |
|---|---|
| Chandan Sreenivasaiah | Data management, annotation, QA, model training (v8, v14), evaluation scripting |
| Vaibhav Bairathi | Pipeline development, inference engine, Flask web application |
| Praveer Jain | Feature engineering, Phase 2 & 3 model experiments |

---

## License

MIT
