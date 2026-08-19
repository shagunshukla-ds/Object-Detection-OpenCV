# Real-Time Object Detection with YOLO & OpenCV

A real-time object detection system built with YOLO and OpenCV, capable of detecting and classifying objects from live video streams, webcam feeds, or static images.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLO](https://img.shields.io/badge/YOLO-v8-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Demo

<!-- Replace with an actual GIF or screenshot of your detector in action -->
![demo](results/images/demo.gif)

## Overview

This project implements a real-time object detection pipeline that:
- Loads a pretrained (or custom-trained) YOLO model
- Processes video streams frame-by-frame using OpenCV
- Draws bounding boxes, class labels, and confidence scores on detected objects
- Supports webcam input, video files, and static images
- Reports inference speed (FPS) for performance benchmarking

## Project Structure

```
object-detection-project/
├── data/
│   ├── raw/                # Original, unprocessed data
│   └── processed/          # Preprocessed / annotated data
├── models/                 # Saved model weights (.pt, .weights, .onnx)
├── notebooks/
│   └── exploration.ipynb   # EDA and experimentation
├── results/
│   ├── images/              # Output detection images
│   └── videos/              # Output detection videos
├── src/
│   ├── detect.py            # Main detection script
│   ├── train.py              # Model training script
│   ├── utils.py              # Helper functions (drawing, preprocessing, metrics)
│   └── config.py             # Configuration and hyperparameters
├── tests/
│   └── test_utils.py        # Unit tests
├── docs/
│   └── architecture.md      # Notes on model architecture / design decisions
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/object-detection-project.git
cd object-detection-project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model weights
```bash
# Example for YOLOv8 (pretrained on COCO)
# Downloaded automatically on first run, or manually via ultralytics
```

## Usage

**Detect objects in a webcam feed:**
```bash
python src/detect.py --source 0
```

**Detect objects in a video file:**
```bash
python src/detect.py --source path/to/video.mp4
```

**Detect objects in an image:**
```bash
python src/detect.py --source path/to/image.jpg
```

**Train on a custom dataset:**
```bash
python src/train.py --data data/processed/data.yaml --epochs 50
```

## Results

| Metric | Value |
|---|---|
| mAP@0.5 | -- |
| Precision | -- |
| Recall | -- |
| Inference Speed | -- FPS |

<!-- Fill in with your actual benchmark numbers -->

## Tech Stack

- **Python 3.9+**
- **OpenCV** — video/image I/O and frame processing
- **YOLO (Ultralytics)** — object detection model
- **NumPy / Pandas** — data handling
- **Matplotlib** — visualization

## Future Improvements

- [ ] Fine-tune on a custom dataset
- [ ] Add object tracking (e.g., DeepSORT)
- [ ] Deploy as a web app (Flask/FastAPI + Streamlit)
- [ ] Optimize for edge devices (ONNX / TensorRT export)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Shagun**
[LinkedIn](#) • [GitHub](#) • [Portfolio](#)
