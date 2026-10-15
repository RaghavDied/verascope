# VeraScope &mdash; Explainable AI Deepfake Verification Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **VeraScope** is an AI-powered multi-modal deepfake detection and forensic evidence platform. It identifies manipulated images and videos with high confidence, produces interpretable visual heatmaps via Grad-CAM, and outputs downloadable forensic verification reports (PDF).

---

## 📌 Problem Statement

The rapid proliferation of Generative AI has facilitated the generation of hyper-realistic synthetic media (deepfakes, face swaps, neural re-enactments). Most existing deepfake detectors operate as single-modality black-box classifiers without verifiable evidence or interpretability. 

**VeraScope** bridges this trust gap by combining:
1. **Multi-Modal Forensics**: Analyzes both static images (`.jpg`, `.png`, `.jpeg`) and temporal videos (`.mp4`, `.avi`, `.mov`).
2. **Strict Face Isolation**: Automatically crops facial regions using MTCNN to eliminate background noise and focus solely on biometric manipulation boundaries.
3. **Deep Convolutional Classification**: Employs an EfficientNet-B4 backbone fine-tuned on benchmark deepfake datasets.
4. **Explainable AI (XAI)**: Generates Grad-CAM neural activation heatmaps that highlight suspicious facial regions (e.g. eyes, mouth, blending borders).
5. **Downloadable Forensic Reports**: Compiles verification findings, statistical breakdowns, and visual evidence into official PDF reports.

---

## 🏗️ Architecture & Pipeline

```text
  [ Upload Image / Video ]
              │
              ▼
   [ Format Validation ]
              │
      ┌───────┴───────────────────────┐
      │ (Image)                       │ (Video)
      ▼                               ▼
[ MTCNN Face Crop ]          [ Temporal Frame Sampling (12 frames) ]
      │                               │
      │                               ▼
      │                      [ MTCNN Face Crop per Frame ]
      │                               │
      └───────────────┬───────────────┘
                      │
                      ▼
        [ EfficientNet-B4 Model ]
    (380×380 resolution, ImageNet Norm)
                      │
                      ▼
          [ Sigmoid Probability ]
                      │
                      ├──────────────────────────┐
                      ▼                          ▼
          [ Grad-CAM Visual Heatmap ]    [ Verdict & Confidence ]
                      │                          │
                      └───────────┬──────────────┘
                                  ▼
                     [ Forensic PDF Report ]
                                  │
                                  ▼
               [ Interactive Web UI & Download ]
```

---

## 📊 Model Evaluation Performance

Evaluated against standard deepfake benchmark datasets:

| Dataset / Domain | Images Evaluated | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **140k Real/Fake Test** | 2,000 | **99.40%** | 99.70% | 99.10% | 0.994 | **0.9999** |
| **FaceForensics++ (FF++)** | 1,275 | **82.12%** | 83.23% | 61.78% | 0.709 | **0.8609** |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (Recommended Python 3.11)
- CUDA-compatible GPU (Optional; CPU fallback is fully supported)

### 2. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/RaghavDied/verascope.git
cd verascope
pip install -r requirements.txt
```

### 3. Model Weights
Place the trained EfficientNet-B4 checkpoint `effnet_b4_deepfake.pt` in the `models/` directory:
```text
models/
  └── effnet_b4_deepfake.pt
```

### 4. Running the Application
Launch the Flask development server:
```bash
python app/app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Running Integration Tests
Execute the end-to-end automated test suite:
```bash
python -m unittest tests/test_pipeline.py
```
The test suite validates:
- Model loading and CPU/CUDA device management
- Image inference with MTCNN face cropping and Grad-CAM generation
- Multi-frame video inference and temporal scoring
- Forensic PDF report generation
- Flask web routes and multipart upload endpoints

---

## 📁 Repository Structure

```text
verascope/
├── app/
│   ├── templates/
│   │   ├── base.html            # Responsive UI layout and styling
│   │   ├── index.html           # Upload dropzone and interactive preview
│   │   └── result.html          # Forensic verdict, XAI heatmaps, video breakdown
│   ├── app.py                   # Flask server and route orchestration
│   ├── inference.py             # Image inference, MTCNN cropping, and Grad-CAM
│   ├── report.py                # ReportLab forensic PDF report generator
│   └── video_inference.py       # Multi-frame video sampling and aggregation
├── data_prep/
│   ├── extract_frames.py        # Video frame extraction utility
│   └── prepare_dataset.py       # Dataset preprocessing and splitting
├── explainability/
│   └── gradcam.py               # Grad-CAM heatmap implementation
├── models/
│   └── effnet_b4_deepfake.pt   # EfficientNet-B4 checkpoint weights
├── tests/
│   └── test_pipeline.py         # End-to-end test suite
├── training/
│   └── train_image_model.py     # EfficientNet-B4 training pipeline
├── final_evaluation.csv         # Benchmark evaluation metrics
├── MODEL_HANDOFF.md             # Model architecture & integration reference
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## 👥 Authors & Team
- **Raghav** &mdash; Frontend UI, Flask Backend, System Integration, PDF Reporting
- **Ishani** &mdash; Deep Learning Architecture, Dataset Preparation, Model Training, Grad-CAM Explainability
