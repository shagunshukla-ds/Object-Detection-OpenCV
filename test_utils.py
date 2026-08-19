"""
test_utils.py

Basic unit tests for utility functions.
Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from utils import compute_iou, get_fps_text


def test_compute_iou_identical_boxes():
    box = [0, 0, 10, 10]
    assert compute_iou(box, box) == 1.0


def test_compute_iou_no_overlap():
    box_a = [0, 0, 5, 5]
    box_b = [10, 10, 15, 15]
    assert compute_iou(box_a, box_b) == 0.0


def test_compute_iou_partial_overlap():
    box_a = [0, 0, 10, 10]
    box_b = [5, 5, 15, 15]
    iou = compute_iou(box_a, box_b)
    assert 0 < iou < 1


def test_get_fps_text():
    assert get_fps_text(30.0) == "FPS: 30.0"
