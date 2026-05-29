"""
dataset.py
----------
Train/val/test split, augmentation-aware Dataset, and DataLoader factory.
"""

import os
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import (
    CROP_PATH, IMAGE_SIZE, BATCH_SIZE, NUM_AUG,
    NUM_WORKERS, VAL_SIZE, TEST_SIZE, RANDOM_STATE,
)

# ── Transforms ────────────────────────────────────────────────────────
_geo_aug = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

# Colour jitter is image-only; applying it to masks corrupts binary labels.
_color_aug = transforms.ColorJitter(brightness=0.1, contrast=0.1)

_base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
_to_tensor = transforms.ToTensor()


# ── Split helper ──────────────────────────────────────────────────────

def train_val_test_split(images, masks, val_size=VAL_SIZE, test_size=TEST_SIZE,
                         random_state=RANDOM_STATE):
    """
    Splits paired image/mask lists into train, val, and test subsets
    while keeping correspondence.

    Returns (train_imgs, val_imgs, test_imgs, train_masks, val_masks, test_masks).
    """
    assert len(images) == len(masks), "images and masks must have the same length."

    if random_state is not None:
        np.random.seed(random_state)

    n       = len(images)
    val_n   = math.ceil(val_size  * n)
    test_n  = math.ceil(test_size * n)
    train_n = n - val_n - test_n

    idx       = np.random.permutation(n)
    train_idx = idx[:train_n]
    val_idx   = idx[train_n:train_n + val_n]
    test_idx  = idx[train_n + val_n:]

    return (
        [images[i] for i in train_idx], [images[i] for i in val_idx],  [images[i] for i in test_idx],
        [masks[i]  for i in train_idx], [masks[i]  for i in val_idx],  [masks[i]  for i in test_idx],
    )


# ── Dataset ───────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """
    Returns (image, mask) tensor pairs.

    When `augment=True`, each original image generates `num_aug` additional
    augmented copies so that __len__ == N * (num_aug + 1).  Geometry
    transforms are applied identically to both image and mask via a shared
    random seed.  Colour jitter is applied to the image only.
    """

    def __init__(self, image_files, mask_files, img_dir, mask_dir,
                 augment=False, num_aug=NUM_AUG):
        self.image_files = image_files
        self.mask_files  = mask_files
        self.img_dir     = img_dir
        self.mask_dir    = mask_dir
        self.augment     = augment
        self.num_aug     = num_aug if augment else 0

    def __len__(self):
        return len(self.image_files) * (self.num_aug + 1)

    def __getitem__(self, idx):
        orig_idx = idx // (self.num_aug + 1)
        aug_idx  = idx %  (self.num_aug + 1)

        image = Image.open(os.path.join(self.img_dir,  self.image_files[orig_idx])).convert("L")
        mask  = Image.open(os.path.join(self.mask_dir, self.mask_files[orig_idx])).convert("L")

        if aug_idx == 0 or not self.augment:
            image = _base_transform(image)
            mask  = _base_transform(mask)
        else:
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed); image = _geo_aug(image)
            torch.manual_seed(seed); mask  = _geo_aug(mask)
            image = _color_aug(image)
            image = _to_tensor(image)
            mask  = _to_tensor(mask)

        return image, mask

    def show_sample(self, orig_idx=0):
        """Display one original image/mask pair and all its augmented versions."""
        cols = self.num_aug + 1
        fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6))
        titles = ["Original"] + [f"Aug {i}" for i in range(1, cols)]

        for aug_idx in range(cols):
            img, msk = self[orig_idx * cols + aug_idx]
            axes[0, aug_idx].imshow(img.squeeze(), cmap="gray")
            axes[0, aug_idx].set_title(titles[aug_idx])
            axes[0, aug_idx].axis("off")
            axes[1, aug_idx].imshow(msk.squeeze(), cmap="gray")
            axes[1, aug_idx].axis("off")

        axes[0, 0].set_ylabel("Image", fontsize=12)
        axes[1, 0].set_ylabel("Mask",  fontsize=12)
        plt.suptitle("Original & Augmented Samples")
        plt.tight_layout()
        plt.show()


# ── DataLoader factory ────────────────────────────────────────────────

def build_dataloaders(crop_path=CROP_PATH):
    """
    Loads cropped images, splits into train/val/test, and returns three
    DataLoaders plus the file lists and directory paths.
    """
    image_dir = os.path.join(crop_path, "images")
    mask_dir  = os.path.join(crop_path, "masks")

    all_images = sorted(os.listdir(image_dir))
    all_masks  = sorted(os.listdir(mask_dir))

    train_imgs, val_imgs, test_imgs, train_masks, val_masks, test_masks = \
        train_val_test_split(all_images, all_masks)

    print(f"Train: {len(train_imgs)}  |  Val: {len(val_imgs)}  |  Test: {len(test_imgs)}")

    train_dataset = BrainTumorDataset(train_imgs, train_masks, image_dir, mask_dir,
                                       augment=True,  num_aug=NUM_AUG)
    val_dataset   = BrainTumorDataset(val_imgs,   val_masks,   image_dir, mask_dir,
                                       augment=False)
    test_dataset  = BrainTumorDataset(test_imgs,  test_masks,  image_dir, mask_dir,
                                       augment=False)

    print(f"Train samples: {len(train_dataset)}  |  "
          f"Val: {len(val_dataset)}  |  Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return (train_loader, val_loader, test_loader,
            train_imgs, val_imgs, test_imgs,
            train_masks, val_masks, test_masks,
            image_dir, mask_dir)