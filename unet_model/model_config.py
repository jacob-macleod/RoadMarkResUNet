"""
Define the configuration for the model
"""

import torch

MODEL_CONFIG = {
    "root_dir": "ROAD MARK.v4i.coco",
    "img_size": (256, 256),  # Resize input for U-Net (must be divisible by 16)
    "batch_size": 8,
    "learning_rate": 2e-3,
    "epochs": 25,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
