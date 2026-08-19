# Architecture & Design Notes

## Pipeline Overview

1. **Input** — Frame captured via OpenCV (`cv2.VideoCapture`) from webcam, video file, or a static image.
2. **Inference** — Frame passed to a YOLO model (Ultralytics implementation), which outputs bounding boxes, class IDs, and confidence scores.
3. **Post-processing** — Non-max suppression (handled internally by YOLO) filters overlapping boxes; results below the confidence threshold are discarded.
4. **Visualization** — Bounding boxes and labels are drawn on the frame using OpenCV drawing utilities.
5. **Output** — Annotated frame is displayed live and/or written to disk.

## Why YOLO?

YOLO (You Only Look Once) frames object detection as a single regression problem — predicting bounding boxes and class probabilities directly from full images in one pass. This makes it significantly faster than two-stage detectors (e.g., R-CNN family), which is essential for real-time applications.

## Design Decisions

- **Ultralytics YOLOv8** was chosen for its actively maintained Python API, ONNX/TensorRT export support, and strong pretrained COCO weights.
- **OpenCV** handles all I/O and visualization since it's lightweight and has native video capture support across platforms.
- **Modular structure** (`detect.py`, `train.py`, `utils.py`, `config.py`) keeps inference, training, and helper logic separate for easier testing and extension.

## Possible Extensions

- Swap in a custom-trained model for domain-specific classes (e.g., PPE detection, retail shelf monitoring).
- Add object tracking (e.g., ByteTrack/DeepSORT) to maintain object identity across frames.
- Export to ONNX/TensorRT for deployment on edge devices (Jetson Nano, Raspberry Pi + Coral).
