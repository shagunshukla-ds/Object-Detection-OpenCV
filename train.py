"""
train.py

Fine-tune a YOLO model on a custom dataset.

Usage:
    python train.py --data data/processed/data.yaml --epochs 50 --batch 16
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on a custom dataset")
    parser.add_argument("--data", type=str, required=True,
                         help="Path to data.yaml describing the dataset")
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                         help="Base weights to start training from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="exp")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )

    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
