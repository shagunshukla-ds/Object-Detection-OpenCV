"""
config.py

Centralized configuration for model and pipeline settings.
"""

MODEL_WEIGHTS = "models/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

CLASS_COLORS = {
    # Optional: assign fixed colors per class ID for consistent visualization
    # 0: (255, 0, 0),
    # 1: (0, 255, 0),
}

RESULTS_DIR = "results"
DATA_DIR = "data"
