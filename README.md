# RoadMarkResUNet

An optimised residual U-Net model with channel attention for real-time road marking binary semantic segmentation. Published in the FSEM 2026 conference at University of Warwick

See https://youtu.be/cEDjUxaUZNw for a video demo of the model

## Abstract

Reliable road marking segmentation is critical for environment perception in Advanced Driver Assistance Systems (ADAS) but deploying deep learning models into embedded or low-power systems is constrained by computational capabilities. This paper proposes RoadMarkResUNet, a CNN architecture which combines Squeeze-and-Excitation blocks with residual connections. A systematic ablation survey was conducted to evaluate the architecture, demonstrating that the proposed architecture provides performance improvements and a reduction in inference time compared to a baseline U-Net model. The peak performance was achieved by RoadMarkResUNet with an IoU of 0.628 and an F1 Score of 0.747, while maintaining low inference times, providing a blueprint for balancing segmentation accuracy with resource constraints in real-time vision applications

## Installation

- Clone the repository
- Run `pip install -r requirements.txt`

## Training the Model

This repository contains 4 models, and 4 files to train each model:

- **The baseline U-Net model**: `unet_main.py`
- **U-Net with residual connections added**: `unet_residual_main.py`
- **U-Net with SE blocks added**: `unet_se_block_main.py`
- **The full RoadMarkResUNet model added, with residual connections and SE blocks**: `resunet_main.py`

To train each model, run the corrosponding python file with `python3 <file name>`

## Other

Internally in the code, `RoadMarkResUNet` is referred to as `Mobile_ResUNet` - this is an older name which refers to the same model

Some helper scripts were scaffolded with the assistance of AI generation tools and manually verified for logical and mathematical correctness
