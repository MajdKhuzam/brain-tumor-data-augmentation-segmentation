import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image

from config import (
    IMAGE_DIR, IMG_WIDTH, IMG_HEIGHT,
    BATCH_SIZE, AUGMENTATION_FACTOR,
)


def augment_image(image: Image.Image) -> Image.Image:
    """Apply random horizontal flip and slight rotation."""
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    angle_deg = random.uniform(-0.2, 0.2) * (180 / np.pi)
    image = image.rotate(angle_deg, resample=Image.BILINEAR)

    return image


def preprocess_to_numpy(image: Image.Image) -> np.ndarray:
    """Resize, add channel dim, and normalise to [-1, 1]."""
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=-1)   # (H, W) → (H, W, 1)
    return (img_array / 127.5) - 1.0


def load_images(image_dir: str = IMAGE_DIR) -> np.ndarray:
    """
    Load every image from *image_dir*, create AUGMENTATION_FACTOR-1
    augmented copies of each, and return a shuffled numpy array.
    """
    images = []
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(image_dir, filename)
        try:
            original_img = Image.open(image_path).convert('L')
            images.append(preprocess_to_numpy(original_img))

            for _ in range(AUGMENTATION_FACTOR - 1):
                aug_img = augment_image(original_img)
                images.append(preprocess_to_numpy(aug_img))
        except Exception as e:
            print(f"Warning: could not process {image_path}: {e}")

    images = np.array(images)
    np.random.shuffle(images)
    print(f"Total images in dataset: {len(images)}")
    return images


def build_dataset(images: np.ndarray) -> tf.data.Dataset:
    """Wrap a numpy array in a cached, shuffled, batched tf.data.Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.cache() # This loads the dataset into RAM during the first epoch
    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset