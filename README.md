# Invigilo — AI Exam Proctoring System

> AI-assisted suspicious behaviour detection in physical exam rooms using computer vision.
> Built as a university capstone project at UTS Sydney (Autumn 2026).

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20%26%20Gaze-orange)]()

## Problem

Traditional exam invigilation relies on human supervisors who cannot monitor every student
simultaneously. Invigilo uses a multi-stage computer vision pipeline to flag suspicious
behaviours in real time, reducing reliance on human attention alone.

## System Architecture — 4 Pipeline Stages

| Stage | What it does | Models used |
|---|---|---|
| A — Student Detection & Seat Tracking | Detects students and assigns them to exam seats using polygon zoning | YOLOv8 |
| B — Prohibited Object Detection | Detects phones, notes, and hand-object interactions | YOLOv8 (custom-trained) |
| C — Head Pose & Gaze Tracking | Estimates head orientation and gaze direction using 468 facial landmarks | MediaPipe FaceMesh · 6DRepNet |
| D — Behaviour Classification | Classifies short video clips as suspicious or normal using temporal models | 3D-CNN · BiLSTM |

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Pose & Gaze:** MediaPipe (468 facial landmarks), 6DRepNet
- **Temporal modelling:** 3D-CNN, BiLSTM
- **Image processing:** OpenCV, Python
- **Development:** Jupyter Notebooks, Python scripts

## Repository Structure
Invigilo/
├── phase1_dataExtraction_processing_scripts/   Data collection and preprocessing
├── phase2_feature_extraction_notebooks/        Feature extraction per pipeline stage
├── phase3_training_notebooks/                  Model training experiments
├── phase4_inference/                           Inference and detection scripts
├── phase5_GUI/                                 GUI for live demo
├── model/                                      Trained model weights
├── data/05_annotations_csv/                    Annotation files
└── docs/                                       Design documents and reports

## My Contributions — Chandan Sreenivasaiah

My responsibilities spanned data management, ground truth annotation, quality assurance, model training, and evaluation.

**Dataset & Annotation**
- Created and maintained `clip_labels_master.csv` — the single source of truth for all 393 clips, tracking filename, behaviour class, target seat, scenario ID, and usability status across the full recording lifecycle
- Identified and resolved a shortcut learning problem where resolution differences across scenario groups caused the model to classify by aspect ratio distortion rather than behaviour; verified camera settings per scenario and corrected metadata so resolution normalisation could be applied
- Created `long_video_event_log.csv` — frame-by-frame ground truth annotations for all 9 test videos (107 suspicious events + 36 hard-negative events), including precise start/end times, seat IDs, and behaviour classes
- Created `manual_active_segments.csv` marking the active 4–6 second behaviour windows within each training clip, used by Phase 2 to focus feature extraction on relevant windows rather than treating all 10 seconds equally

**Quality Assurance**
- Managed `target_person_review.csv`; reviewed hundreds of pose visualisation images to verify correct target person identification per clip
- Identified and excluded 130 training windows where the target person was misidentified, occluded, or the behaviour was ambiguous — directly improving classifier signal quality

**Model Training**
- Trained the **v8 baseline experiment** — the first working two-stage MLP with all 5 original suspicious classes (75.7% test accuracy, Macro F1 0.474); v8 results revealed that 3 of 5 classes had recall below 27%, directly motivating the class-merging strategy in v9 onwards
- Trained the **v14 Bidirectional GRU model** — 2-layer BiGRU with 64 hidden units per direction (128 total), 113K parameters; achieved the highest macro F1 of any experiment (0.722) and 97% recall on `looking_down`, demonstrating that temporal dependencies in both directions are key to distinguishing genuine suspicious glances from normal writing posture

**Evaluation**
- Wrote `evaluate_inference.py` — the evaluation script that matches model predictions against ground-truth event logs and computes detection rates, flagging rates, class accuracy, false alarm rates, detection latency, per-person breakdowns, temporal heatmaps, and summary charts
- Ran all evaluations across all three pipeline versions (v1, v2, v3) on the 9 test videos and analysed results to guide each iteration of pipeline improvements

## Team

Built as a university project for Deep Learning & CNN (42028) at the University of Technology Sydney, May 2026.

| Member | Contribution |
|---|---|
| Chandan Sreenivasaiah | Data management, annotation, QA, model training (v8, v14), evaluation scripting |
| Vaibhav Bairathi | Pipeline development, inference engine, Flask web application |
| Praveer Jain | Feature engineering, Phase 2 & 3 model experiments |

## License

MIT
