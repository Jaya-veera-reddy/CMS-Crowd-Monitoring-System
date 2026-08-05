
## Crowd Analytics & Predictive Monitoring System


A hybrid computer vision and predictive analytics platform designed to monitor spatial crowd density in real time, generate short-term trend forecasts, and evaluate crowd management metrics through an interactive web dashboard.

---

## Overview

Crowd monitoring plays a crucial role in ensuring public safety in transportation hubs, stadiums, shopping malls, concerts, and smart city environments. Traditional object detection methods often struggle in highly congested scenes due to severe occlusions where individual people cannot be reliably detected.

This project combines multiple deep learning approaches to overcome these limitations by integrating object detection, density estimation, temporal forecasting, and interactive visualization into a single end-to-end system.

The complete pipeline consists of four major components:

1. **Object Detection (YOLOv8)**  
   Performs fast real-time person detection using bounding boxes. It provides accurate headcounts in low- and medium-density environments while generating live crowd statistics and alerts.

2. **Density Map Estimation (CSRNet)**  
   Utilizes a VGG-16 backbone with dilated convolution layers to generate continuous density maps. Instead of detecting each individual person, CSRNet estimates crowd density spatially, making it highly effective in heavily congested scenes.

3. **Temporal Trend Forecasting**  
   Converts historical crowd counts into time-series sequences to forecast crowd sizes approximately **5 seconds into the future** using **LSTM Neural Networks** and **Moving Average** forecasting.

4. **Interactive Flask Dashboard**  
   A web application (`app.py`) allowing users to upload surveillance videos, trigger pipeline processing, display current and predicted headcounts, generate crowd alerts, and monitor evaluation metrics.

---

## Features

- Real-time crowd counting and person detection
- Continuous crowd density map visualization
- Short-term temporal crowd prediction
- Automated crowd alert generation
- Interactive Flask web dashboard
- Evaluation telemetry and benchmark visualization
- Automated graph and report generation
- ShanghaiTech dataset evaluation support
- Modular and well-organized project architecture

---

## Repository Architecture

```text
crowd-management-system/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── dashboard/
│   ├── templates/
│   └── static/
│       ├── uploads/
│       └── outputs/
│
├── models/
│   ├── csrnet.py
│   └── lstm_predictor.py
│
├── preprocessing/
│   ├── preprocess_shanghai.py
│   └── generate_timeseries.py
│
├── training/
│   ├── train_csrnet.py
│   ├── train_lstm.py
│   └── evaluate_csrnet.py
│
├── inference/
│   ├── yolo_count.py
│   ├── predict_future.py
│   └── extract_video_counts.py
│
├── evaluation/
│   └── evaluate_model.py
│
├── scripts/
│   ├── generate_graphs.py
│   └── generate_report.py
│
├── sample_data/
│   ├── sample_video.mp4
│   └── evaluation_results.jpg
│
├── checkpoints/
├── dataset/
└── .gitignore
```

---

## System Pipeline

```text
Input Video
     │
     ▼
YOLOv8 Person Detection
     │
     ▼
CSRNet Density Estimation
     │
     ▼
Crowd Count Generation
     │
     ▼
Time-Series Creation
     │
     ▼
LSTM / Moving Average Prediction
     │
     ▼
Alert Generation
     │
     ▼
Flask Dashboard Visualization
```

---

## Technologies Used

| Category | Technology |
|-----------|------------|
| Programming Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Object Detection | YOLOv8 |
| Density Estimation | CSRNet (Dilated CNN) |
| Forecasting | LSTM Neural Network |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Web Framework | Flask |
| Dataset | ShanghaiTech Crowd Counting Dataset |

---

## System Performance & Model Metrics

The system continuously evaluates crowd counting accuracy, density estimation quality, and short-term forecasting performance across full evaluation runs.

### 1. Classification & System Performance

Evaluated across crowd alert thresholds and system validation samples (**Processed Samples: 182**).

| Metric | Score / Value | Description |
|--------|---------------|-------------|
| Detection Confidence Cutoff | **0.40** | Bounding box threshold for person tracking |
| Classification Accuracy | **75.27%** | Multi-tier crowd alert classification |
| Weighted Precision | **0.9856** | High-precision rate for active crowd alerts |
| Weighted Recall | **0.7611** | True positive detection rate |
| Weighted F1 Score | **0.8589** | Overall balanced classification score |

---

### 2. Spatial Density Estimation & Forecasting (Regression)

Regression metrics tracking headcount error against ground truth evaluations.

| Metric | Score / Value | Description |
|--------|---------------|-------------|
| Mean Absolute Error (MAE) | **470.65** | Mean absolute headcount deviation |
| Root Mean Squared Error (RMSE) | **687.74** | Root mean squared headcount error |

> **Note:** Raw evaluation outputs, sample video clips (`sample_video.mp4`), and output logs are stored inside the **sample_data/** directory (see **sample_data/evaluation_results.jpg**).

---

# Quick Start

## 1. Installation

Clone the repository:

```bash
git clone https://github.com/your-username/crowd-management-system.git

cd crowd-management-system
```

Create and activate a virtual environment.

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## 2. Running the Flask Dashboard

Launch the Flask application.

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

Upload **sample_data/sample_video.mp4** to automatically execute the crowd analytics pipeline.

---

## 3. Command-Line Usage

### Run YOLO Detection

```bash
python inference/yolo_count.py
```

### Generate Future Crowd Prediction

```bash
python inference/predict_future.py
```

### Evaluate System Telemetry

```bash
python evaluation/evaluate_model.py
```

### Train CSRNet

```bash
python training/train_csrnet.py
```

### Generate Graphs

```bash
python scripts/generate_graphs.py
```

### Generate Reports

```bash
python scripts/generate_report.py
```

---

## Pretrained Model Weights

Due to GitHub's file size limitations, pretrained deep learning model checkpoints are **not included** in this repository.

Download the following files from the repository's **Releases** page:

- `csrnet.pth`
- `lstm_model.pth`

Place them inside the following directory:

```text
checkpoints/
├── csrnet.pth
└── lstm_model.pth
```

---

## Future Enhancements

- Multi-camera crowd fusion
- Real-time RTSP CCTV streaming integration
- GPU acceleration using TensorRT
- Transformer-based temporal crowd forecasting
- Edge AI deployment on NVIDIA Jetson devices

