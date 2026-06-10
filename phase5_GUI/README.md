# Invigilo — AI-Assisted Exam Proctoring System

Real-time suspicious behaviour detection for examination halls using pose-based deep learning.

## Quick Start

### 1. Place files together

```
exam_proctoring_dataset/
├── app.py                        ← This file
├── phase4_inference_v3.py        ← Inference pipeline (required)
├── inference_package_v11.pth     ← Trained model (required)
├── templates/                    ← HTML templates
├── static/css/                   ← Stylesheet
├── uploads/                      ← Created automatically
└── output/                       ← Created automatically
```

### 2. Install

```bash
exam_env\Scripts\activate
pip install flask
```

### 3. Run

```bash
python app.py
```

Open **http://localhost:5000**

### Options

```bash
python app.py --port 5001
python app.py --pkg path/to/model.pth
```

## Features

- **Upload Video** — Drag-and-drop exam footage for analysis
- **Live Webcam** — Monitor a camera feed in real time
- **Live Monitoring** — Watch annotated frames stream with real-time flagged events and event log
- **Start/Stop Control** — Begin and end monitoring on demand
- **Results Dashboard** — Stats, per-person breakdown, downloadable outputs
- **Production Design** — Dark surveillance aesthetic, responsive layout
