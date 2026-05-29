"""
preprocessing.py
----------------
One-time step: crops each MRI image + mask to the bounding box of the
largest external contour and saves the results to CROP_PATH.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset

from config import DATASET_PATH, CROP_PATH


class TumorDataset(Dataset):
    """
    Reads raw MRI images and masks, crops each image to the bounding box
    of the largest external contour (removing background padding), and
    saves the results to `save_dir`.  Used once as a preprocessing step.
    """

    def __init__(self, root_dir, save_dir, apply_cropping=True):
        self.root_dir       = root_dir
        self.save_dir       = save_dir
        self.apply_cropping = apply_cropping
        self.data           = []  # (img_path, mask_path, save_img_path, save_mask_path)

        img_dir       = os.path.join(root_dir, "images")
        mask_dir      = os.path.join(root_dir, "masks")
        save_img_dir  = os.path.join(save_dir,  "images")
        save_mask_dir = os.path.join(save_dir,  "masks")

        os.makedirs(save_img_dir,  exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)

        img_files = sorted(os.listdir(img_dir))
        print(f"Images found: {len(img_files)}  |  "
              f"Masks found: {len(sorted(os.listdir(mask_dir)))}")

        for img_name in img_files:
            img_path  = os.path.join(img_dir,       img_name)
            mask_path = os.path.join(mask_dir,      img_name)
            save_img  = os.path.join(save_img_dir,  img_name)
            save_mask = os.path.join(save_mask_dir, img_name)

            if os.path.exists(mask_path):
                self.data.append((img_path, mask_path, save_img, save_mask))
            else:
                print(f"Warning: no mask found for {img_name}")

    def _crop(self, img, mask):
        """Crop both image and mask to the bounding box of the largest contour."""
        _, thresh = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            img  = img [y:y+h, x:x+w]
            mask = mask[y:y+h, x:x+w]
        return img, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, save_img_path, save_mask_path = self.data[idx]

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Warning: could not load {img_path}")
            return None

        if self.apply_cropping:
            img, mask = self._crop(img, mask)

        cv2.imwrite(save_img_path,  img)
        cv2.imwrite(save_mask_path, mask)

        img_tensor  = torch.tensor(img,  dtype=torch.float32).unsqueeze(0) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        return img_tensor, mask_tensor


def run_preprocessing(dataset_path=DATASET_PATH, crop_path=CROP_PATH):
    """Crop all images and masks; safe to re-run (skips nothing — overwrites)."""
    print("Dataset exists:", os.path.exists(dataset_path))

    crop_dataset = TumorDataset(root_dir=dataset_path, save_dir=crop_path, apply_cropping=True)

    for i in range(len(crop_dataset)):
        if crop_dataset[i] is not None:
            print(f"Processed {i+1}/{len(crop_dataset)}", end="\r")

    print("\nCropping complete!")
    print(f"Saved images: {len(os.listdir(os.path.join(crop_path, 'images')))}  |  "
          f"Saved masks: {len(os.listdir(os.path.join(crop_path, 'masks')))}")


if __name__ == "__main__":
    run_preprocessing()