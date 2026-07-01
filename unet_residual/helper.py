"""
Helper functions to visualise and train
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import time

from unet_residual.model_config import MODEL_CONFIG
from unet_residual.road_mark_dataset import RoadMarkDataset
from unet_residual.unet_residual import UnetResidual


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class HybridLoss(nn.Module):
    """
    Combining BCE and Dice loss to be more robust
    """

    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + 1e-6) / (probs.sum() + targets.sum() + 1e-6)
        dice_loss = 1 - dice

        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(logits, targets, criterion, smooth=1e-6):
    """
    Calculate metrics for a batch of predictions.
    Computes Hybrid Loss, IoU, F1 (Dice), and MAE on the GPU.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    # Calculate Loss using the provided HybridLoss instance
    loss = criterion(logits, targets).item()

    # Flatten batch spatial dimensions for stable segmentations metrics
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    probs_flat = probs.view(probs.size(0), -1)

    # Intersection and Union calculations
    intersection = (preds_flat * targets_flat).sum(dim=1)
    total_pixels = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    # IoU (Jaccard Index)
    union = total_pixels - intersection
    iou = ((intersection + smooth) / (union + smooth)).mean().item()

    # F1 Score (Dice Coefficient)
    f1 = ((2.0 * intersection + smooth) / (total_pixels + smooth)).mean().item()

    # Mean Absolute Error (MAE)
    mae = torch.abs(probs_flat - targets_flat).mean().item()

    return loss, iou, f1, mae


def visualize_prediction(model, dataset, index=0):
    model.eval()
    with torch.no_grad():
        image, mask = dataset[index]
        image_tensor = image.unsqueeze(0).to(MODEL_CONFIG["device"])
        output = model(image_tensor)
        pred_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred_mask = (pred_prob > 0.5).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        img_np = image.permute(1, 2, 0).numpy()
        ax[0].imshow(img_np)
        ax[0].set_title("Input")
        ax[1].imshow(mask[0], cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[2].imshow(pred_mask, cmap="gray")
        ax[2].set_title("Mobile-ResUNet")
        plt.show()


def train_model():
    print(f"Initializing Mobile-ResUNet on {MODEL_CONFIG['device']}...")
    g = torch.Generator()
    g.manual_seed(42)

    # Instantiate Datasets without transforms
    train_dataset = RoadMarkDataset(MODEL_CONFIG["root_dir"], subfolder="train")
    valid_dataset = RoadMarkDataset(MODEL_CONFIG["root_dir"], subfolder="valid")
    test_dataset = RoadMarkDataset(MODEL_CONFIG["root_dir"], subfolder="test")

    train_loader = DataLoader(
        train_dataset, batch_size=MODEL_CONFIG["batch_size"], shuffle=True, generator=g
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=MODEL_CONFIG["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=MODEL_CONFIG["batch_size"], shuffle=False
    )

    model = UnetResidual(n_channels=3, n_classes=1).to(MODEL_CONFIG["device"])

    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG["learning_rate"])

    total_start_time = time.time()

    # --- TRAINING ---
    for epoch in range(MODEL_CONFIG["epochs"]):
        model.train()
        train_loss, train_iou, train_f1, train_mae = 0, 0, 0, 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(MODEL_CONFIG["device"])
            masks = masks.to(MODEL_CONFIG["device"])

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compile batch metrics
            b_loss, b_iou, b_f1, b_mae = calculate_metrics(outputs, masks, criterion)
            train_loss += b_loss
            train_iou += b_iou
            train_f1 += b_f1
            train_mae += b_mae

        # --- VALIDATION ---
        model.eval()
        val_loss, val_iou, val_f1, val_mae = 0, 0, 0, 0

        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(MODEL_CONFIG["device"])
                masks = masks.to(MODEL_CONFIG["device"])
                outputs = model(images)

                b_loss, b_iou, b_f1, b_mae = calculate_metrics(
                    outputs, masks, criterion
                )
                val_loss += b_loss
                val_iou += b_iou
                val_f1 += b_f1
                val_mae += b_mae

        num_train = len(train_loader)
        num_val = len(valid_loader)

        print(
            f"Epoch [{epoch+1}/{MODEL_CONFIG['epochs']}]\n"
            f"  Train -> Loss: {train_loss/num_train:.4f} | IoU: {train_iou/num_train:.4f} | F1: {train_f1/num_train:.4f} | MAE: {train_mae/num_train:.4f}\n"
            f"  Val   -> Loss: {val_loss/num_val:.4f} | IoU: {val_iou/num_val:.4f} | F1: {val_f1/num_val:.4f} | MAE: {val_mae/num_val:.4f}"
        )

        # if (epoch + 1) % 3 == 0:
        #     visualize_prediction(model, valid_dataset)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Convert total seconds to hours, minutes, and seconds
    hours, rem = divmod(total_duration, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\nTraining Complete.")
    print(
        f"Total Training Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s "
        f"({total_duration:.2f} total seconds)"
    )

    torch.save(model.state_dict(), "mobile_resunet.pth")
    print("Model Saved.")

    # --- FINAL TEST ---
    print("\nRunning Final Test Evaluation...")
    model.eval()
    test_loss, test_iou, test_f1, test_mae = 0, 0, 0, 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(MODEL_CONFIG["device"])
            masks = masks.to(MODEL_CONFIG["device"])
            outputs = model(images)

            b_loss, b_iou, b_f1, b_mae = calculate_metrics(outputs, masks, criterion)
            test_loss += b_loss
            test_iou += b_iou
            test_f1 += b_f1
            test_mae += b_mae

    num_test = len(test_loader)
    print(
        f"FINAL TEST METRICS:\n"
        f"  Loss: {test_loss/num_test:.4f} | IoU: {test_iou/num_test:.4f} | F1: {test_f1/num_test:.4f} | MAE: {test_mae/num_test:.4f}"
    )
