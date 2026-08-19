"""
detect.py

Real-time object detection using YOLO + OpenCV.
Supports webcam, video file, or image input.

Usage:
    python detect.py --source 0                     # webcam
    python detect.py --source path/to/video.mp4      # video file
    python detect.py --source path/to/image.jpg       # image
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import draw_detections, get_fps_text


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + OpenCV Object Detection")
    parser.add_argument(
        "--source", type=str, default="0",
        help="Path to video/image file, or '0' for webcam"
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8n.pt",
        help="Path to YOLO model weights"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save output to results/ directory"
    )
    return parser.parse_args()


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def run_detection(args):
    model = YOLO(args.weights)

    source = 0 if args.source == "0" else args.source

    if isinstance(source, str) and is_image(source):
        frame = cv2.imread(source)
        results = model(frame, conf=args.conf)[0]
        frame = draw_detections(frame, results)
        cv2.imshow("Detection", frame)
        if args.save:
            out_path = Path("results/images") / Path(source).name
            cv2.imwrite(str(out_path), frame)
            print(f"Saved to {out_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter("results/videos/output.mp4", fourcc, 20.0, (w, h))

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]
        frame = draw_detections(frame, results)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, get_fps_text(fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Detection", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_detection(args)
