"""
utils.py
--------
Checkpoint I/O, accuracy / Dice / IoU evaluation, and prediction saving.
"""

import os
import torch
import torch.nn as nn
import torchvision

from config import DEVICE, CHECKPOINT_PATH, SAVED_IMAGES_PATH


def save_checkpoint(state, filename=CHECKPOINT_PATH):
    torch.save(state, filename)
    print(f"  Checkpoint saved → {filename}")


def load_checkpoint(checkpoint, model):
    """
    Loads a checkpoint into `model`, handling both plain and DataParallel
    models.  Checkpoints are always saved from the unwrapped base model
    (no `module.` prefix), so this loads cleanly in either case.
    """
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(checkpoint["state_dict"])
    print("  Checkpoint loaded.")


def check_accuracy(loader, model, loss_fn, device=DEVICE):
    """
    Evaluates model on `loader`.
    Returns (dice, iou, avg_loss) as plain Python floats.
    Metrics are computed globally across all batches (not averaged per-batch).
    """
    model.eval()

    total_loss  = 0.0
    num_correct = 0
    num_pixels  = 0
    total_inter = 0.0
    total_pred  = 0.0
    total_true  = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Binarise masks — guards against interpolation artefacts from Resize
            y_bin = (y > 0.5).float()

            logits = model(x)
            total_loss += loss_fn(logits, y_bin).item()

            preds = (torch.sigmoid(logits) > 0.5).float()

            num_correct += (preds == y_bin).sum().item()
            num_pixels  += preds.numel()

            total_inter += (preds * y_bin).sum().item()
            total_pred  += preds.sum().item()
            total_true  += y_bin.sum().item()

    avg_loss = total_loss  / len(loader)
    accuracy = num_correct / num_pixels * 100
    dice     = (2 * total_inter) / (total_pred + total_true  + 1e-8)
    iou      = total_inter       / (total_pred + total_true  - total_inter + 1e-8)

    print(f"  Accuracy: {accuracy:.2f}%  |  "
          f"Dice: {dice:.4f}  |  "
          f"IoU: {iou:.4f}  |  "
          f"Loss: {avg_loss:.4f}")

    model.train()
    return dice, iou, avg_loss


def save_predictions_as_imgs(loader, model, folder=SAVED_IMAGES_PATH, device=DEVICE):
    """Save predicted and ground-truth mask images to `folder`."""
    os.makedirs(folder, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            preds = (torch.sigmoid(model(x.to(device))) > 0.5).float()
            torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
            torchvision.utils.save_image(y,     f"{folder}/gt_{idx}.png")
    model.train()
    print(f"  Predictions saved to {folder}/")