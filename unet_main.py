"""
Main program flow
"""

import torch
import numpy as np
import random
import os
import gc


def set_seed(seed=42):
    # 1. Python & Numpy seeds
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # 2. PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # 3. Force CuDNN to be deterministic (Note: this may slightly slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Call this before you do anything else!
set_seed(42)

from unet_model.helper import train_model

print("UNET Baseline")
# for i in range(1, 4):
#     print(f"Running iteration {i}")

#     train_model()

#     torch.cuda.empty_cache()
#     gc.collect()
train_model()
