import torch
import os

# ── Paths ──────────────────────────────────────────────────────────────
DATASET_PATH      = os.path.abspath(os.path.join(__file__, '..', 'data'))
OUTPUT_PATH       = os.path.abspath(os.path.join(__file__, '..', 'output'))
os.makedirs(OUTPUT_PATH, exist_ok=True)
CROP_PATH         = os.path.join(OUTPUT_PATH, 'cropped_images')
CHECKPOINT_PATH   = os.path.join(OUTPUT_PATH, 'best_model.pth.tar')
SAVED_IMAGES_PATH = os.path.join(OUTPUT_PATH, 'saved_images')

# ── Hyperparameters ────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS      = torch.cuda.device_count()
IMAGE_SIZE    = 128
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 20
NUM_AUG       = 5
NUM_WORKERS   = 4
LOAD_MODEL    = False

# ── Split ratios ───────────────────────────────────────────────────────
VAL_SIZE     = 0.15
TEST_SIZE    = 0.15
RANDOM_STATE = 42