"""
utils.py

Helper functions for drawing bounding boxes, labels,
and computing simple performance metrics.
"""

import cv2
import numpy as np


def draw_detections(frame, results, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes, class labels, and confidence scores on a frame.

    Args:
        frame: The image/frame (numpy array) to draw on.
        results: A YOLO results object (from ultralytics) for this frame.
        color: Box color in BGR.
        thickness: Line thickness for boxes.

    Returns:
        The annotated frame.
    """
    if results.boxes is None:
        return frame

    names = results.names
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def get_fps_text(fps: float) -> str:
    return f"FPS: {fps:.1f}"


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)

    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0
