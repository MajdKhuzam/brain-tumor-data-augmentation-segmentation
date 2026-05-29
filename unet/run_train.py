"""
main.py
-------
Runs the full pipeline:
  1. Preprocessing (crop images)
  2. Build dataloaders
  3. Build model
  4. Train
  5. Evaluate on test set + plot curves
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE, NUM_GPUS, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, CHECKPOINT_PATH, LOAD_MODEL,
)
from preprocessing import run_preprocessing
from dataset import build_dataloaders
from model import build_model
from train import train
from evaluate import plot_training_curves, evaluate_on_test, visualise_predictions
from utils import load_checkpoint


def main():
    print(f"Device: {DEVICE}  |  GPUs available: {NUM_GPUS}")

    # ── 1. Preprocessing ──────────────────────────────────────────────
    run_preprocessing()

    # ── 2. Data ───────────────────────────────────────────────────────
    (train_loader, val_loader, test_loader,
     train_imgs, val_imgs, test_imgs,
     train_masks, val_masks, test_masks,
     image_dir, mask_dir) = build_dataloaders()

    # Sanity check
    imgs, masks = next(iter(train_loader))
    print(f"Image batch : {imgs.shape}   range [{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"Mask  batch : {masks.shape}  range [{masks.min():.2f}, {masks.max():.2f}]")

    # ── 3. Model ──────────────────────────────────────────────────────
    model = build_model(DEVICE)

    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
        print(f"Using {NUM_GPUS} GPUs via DataParallel.")

    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT_PATH, map_location=DEVICE), model)

    print("Initial validation metrics:")
    from utils import check_accuracy
    check_accuracy(val_loader, model, loss_fn)

    # ── 4. Training ───────────────────────────────────────────────────
    train_losses, val_losses, dice_scores, iou_scores = train(
        model, train_loader, val_loader, optimizer, loss_fn, scaler, NUM_EPOCHS
    )

    # ── 5. Evaluation ─────────────────────────────────────────────────
    plot_training_curves(train_losses, val_losses, dice_scores, iou_scores)
    evaluate_on_test(test_loader, model, loss_fn)
    visualise_predictions()


if __name__ == "__main__":
    main()