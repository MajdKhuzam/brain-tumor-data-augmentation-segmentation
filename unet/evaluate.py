"""
evaluate.py
-----------
Plot training curves and run final evaluation on the test set.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import SAVED_IMAGES_PATH, OUTPUT_PATH
from utils import check_accuracy


def plot_training_curves(
    train_losses=None, val_losses=None,
    dice_scores=None,  iou_scores=None,
):
    """
    Plot loss, Dice, and IoU curves.
    Pass arrays directly, or leave as None to load from the saved .npy files.
    """
    train_losses = train_losses if train_losses is not None else np.load(OUTPUT_PATH + "/train_losses.npy")
    val_losses   = val_losses   if val_losses   is not None else np.load(OUTPUT_PATH + "/val_losses.npy")
    dice_scores  = dice_scores  if dice_scores  is not None else np.load(OUTPUT_PATH + "/dice_scores.npy")
    iou_scores   = iou_scores   if iou_scores   is not None else np.load(OUTPUT_PATH + "/iou_scores.npy")

    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_losses, label="Train", color="steelblue")
    axes[0].plot(epochs, val_losses,   label="Val",   color="tomato")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCEWithLogits")
    axes[0].legend()

    axes[1].plot(epochs, dice_scores, color="mediumseagreen")
    axes[1].set(title="Dice Score", xlabel="Epoch", ylabel="Dice")

    axes[2].plot(epochs, iou_scores, color="mediumpurple")
    axes[2].set(title="IoU Score", xlabel="Epoch", ylabel="IoU")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_on_test(test_loader, model, loss_fn):
    """Run check_accuracy on the test set and return (dice, iou, loss)."""
    print("Test-set evaluation:")
    return check_accuracy(test_loader, model, loss_fn)


def visualise_predictions(n_show=10, saved_images_path=SAVED_IMAGES_PATH):
    """Display ground-truth vs predicted masks side by side."""
    fig, axes = plt.subplots(n_show, 2, figsize=(6, 3 * n_show))
    axes[0, 0].set_title("Ground Truth", fontsize=12)
    axes[0, 1].set_title("Prediction",   fontsize=12)

    for i in range(n_show):
        pred_img = cv2.imread(os.path.join(saved_images_path, f"pred_{i}.png"), cv2.IMREAD_GRAYSCALE)
        gt_img   = cv2.imread(os.path.join(saved_images_path, f"gt_{i}.png"),   cv2.IMREAD_GRAYSCALE)

        if pred_img is None or gt_img is None:
            print(f"Warning: image {i} not found, skipping.")
            continue

        axes[i, 0].imshow(gt_img,   cmap="gray"); axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_img, cmap="gray"); axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()