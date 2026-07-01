"""
Helper functions to visualise and train
"""

import time
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from unet_model.model_config import MODEL_CONFIG
from unet_model.road_mark_dataset import RoadMarkDataset
from unet_model.basic_unet_model import BasicUnetModel


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


def visualise_prediction(model, dataset, index=0):
    """
    Show input, ground truth and prediction

    Args:
        model: The model to use
        dataset: The dataset to use
        index (int, optional): The index to use. Defaults to 0.
    """
    model.eval()
    with torch.no_grad():
        image, mask = dataset[index]
        image_tensor = image.unsqueeze(0).to(MODEL_CONFIG["device"])

        # Forward pass
        output = model(image_tensor)
        pred_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred_mask = (pred_prob > 0.5).astype(np.uint8)

        # Plot
        _, ax = plt.subplots(1, 3, figsize=(12, 4))

        # Valid image for plotting
        img_np = image.permute(1, 2, 0).numpy()

        ax[0].imshow(img_np)
        ax[0].set_title("Input Image")

        ax[1].imshow(mask[0], cmap="gray")
        ax[1].set_title("Ground Truth Mask")

        ax[2].imshow(pred_mask, cmap="gray")
        ax[2].set_title("Predicted Mask")
        plt.show()


def calculate_metrics(logits, targets, criterion, smooth=1e-6):
    """
    Calculate metrics for a batch of predictions.
    Computes Loss, IoU, F1 (Dice), and MAE on the GPU.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    # Calculate Loss
    # loss = criteria(logits, targets).item()
    loss = criterion(logits, targets).item()

    # Flatten spatial dimensions for segmentation metrics
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    probs_flat = probs.view(probs.size(0), -1)

    # Intersection and Union calculation
    intersection = (preds_flat * targets_flat).sum(dim=1)
    total_pixels = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    # IoU (Jaccard Index)
    union = total_pixels - intersection
    iou = ((intersection + smooth) / (union + smooth)).mean().item()

    # F1 Score (Dice Coefficient)
    f1 = ((2.0 * intersection + smooth) / (total_pixels + smooth)).mean().item()

    # Mean Absolute Error (MAE) on probabilities
    mae = torch.abs(probs_flat - targets_flat).mean().item()

    return loss, iou, f1, mae


def train_model():
    """
    Train the model
    """
    print(f"Training on {MODEL_CONFIG['device']}")
    g = torch.Generator()
    g.manual_seed(42)
    train_dataset = RoadMarkDataset(MODEL_CONFIG["root_dir"], subfolder="train")
    valid_dataset = RoadMarkDataset(MODEL_CONFIG["root_dir"], subfolder="valid")

    train_loader = DataLoader(
        train_dataset, batch_size=MODEL_CONFIG["batch_size"], shuffle=True, generator=g
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=MODEL_CONFIG["batch_size"], shuffle=False
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(valid_dataset)}")

    # Initialise the model
    model = BasicUnetModel(n_channels=3, n_classes=1).to(MODEL_CONFIG["device"])

    # Set up the loss and optimiser
    # criteria = torch.nn.BCEWithLogitsLoss()
    criteria = HybridLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=MODEL_CONFIG["learning_rate"])

    total_start_time = time.time()
    for epoch in range(MODEL_CONFIG["epochs"]):
        # --- TRAINING ---
        model.train()
        train_loss, train_iou, train_f1, train_mae = 0, 0, 0, 0

        for images, masks in train_loader:
            images = images.to(MODEL_CONFIG["device"])
            masks = masks.to(MODEL_CONFIG["device"])

            # Forward
            outputs = model(images)
            loss = criteria(outputs, masks)

            # Backward
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Metric compilation
            b_loss, b_iou, b_f1, b_mae = calculate_metrics(outputs, masks, criteria)
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

                b_loss, b_iou, b_f1, b_mae = calculate_metrics(outputs, masks, criteria)
                val_loss += b_loss
                val_iou += b_iou
                val_f1 += b_f1
                val_mae += b_mae

        # Calculate dataset averages
        num_train = len(train_loader)
        num_val = len(valid_loader)

        print(
            f"Epoch [{epoch+1}/{MODEL_CONFIG['epochs']}]\n"
            f"  Train -> Loss: {train_loss/num_train:.4f} | IoU: {train_iou/num_train:.4f} | F1: {train_f1/num_train:.4f} | MAE: {train_mae/num_train:.4f}\n"
            f"  Val   -> Loss: {val_loss/num_val:.4f} | IoU: {val_iou/num_val:.4f} | F1: {val_f1/num_val:.4f} | MAE: {val_mae/num_val:.4f}"
        )

        # Visualize one result every epoch to check progress
        # if (epoch + 1) % 2 == 0:
        #     visualise_prediction(model, valid_dataset, index=0)

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

    # Save Model
    torch.save(model.state_dict(), "unet_road_model.pth")
    print("Model saved to unet_road_model.pth")
