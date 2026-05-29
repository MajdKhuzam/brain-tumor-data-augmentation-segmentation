"""
train.py
--------
Per-epoch training step and the full training loop with best-checkpoint tracking.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import DEVICE, NUM_EPOCHS, CHECKPOINT_PATH, OUTPUT_PATH
from utils import save_checkpoint, check_accuracy, save_predictions_as_imgs


def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=True)

    for data, targets in loop:
        data    = data.to(DEVICE)
        targets = targets.float().to(DEVICE)

        with torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            preds = model(data)
            loss  = loss_fn(preds, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def train(model, train_loader, val_loader, optimizer, loss_fn, scaler,
          num_epochs=NUM_EPOCHS):
    """
    Full training loop.  Saves the checkpoint that achieves the best
    combined (Dice + IoU) / 2 on the validation set.
    """
    train_losses, val_losses, dice_scores, iou_scores = [], [], [], []
    best_avg_score  = -1.0
    best_checkpoint = None

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        dice, iou, val_loss = check_accuracy(val_loader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        dice_scores.append(dice)
        iou_scores.append(iou)

        avg_score = (dice + iou) / 2
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            best_checkpoint = {
                "state_dict": base_model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            }
            save_checkpoint(best_checkpoint)

    print(f"\nTraining complete.  Best Dice+IoU avg: {best_avg_score:.4f}")

    if best_checkpoint:
        save_checkpoint(best_checkpoint, filename=CHECKPOINT_PATH)
    save_predictions_as_imgs(val_loader, model)

    np.save(OUTPUT_PATH + "/train_losses.npy", np.array(train_losses))
    np.save(OUTPUT_PATH + "/val_losses.npy",   np.array(val_losses))
    np.save(OUTPUT_PATH + "/dice_scores.npy",  np.array(dice_scores))
    np.save(OUTPUT_PATH + "/iou_scores.npy",   np.array(iou_scores))

    return train_losses, val_losses, dice_scores, iou_scores