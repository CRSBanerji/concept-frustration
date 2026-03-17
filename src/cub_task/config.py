from pathlib import Path
import hashlib
import time
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = REPO_ROOT / "data" / "CUB_200_2011"
RESULTS_DIR = REPO_ROOT / "results" / "CUB"

# ---- dataset / task
USE_TRAIN_ONLY = False
MAX_PER_CLASS = None
SEED_DATA = 123

# ---- Concepts
CONCEPT_NAMES = [
    "has_upperparts_color::white",
    "has_shape::gull-like",
    "has_wing_pattern::solid",
]

# ---- Embeddings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
BATCH_SIZE_EMB = 64
IMAGE_SIZE = 224

# ---- Folds / training
K_FOLDS = 10
SEED_FOLDS = 777

HIDDEN_BB = 128
EPOCHS_BB = 60

EPOCHS_CBM = 30
LAMBDA_C = 1.0

K_SAE = 300
EPOCHS_SAE = 60

# ---- Fisher keep window
P_LO, P_HI = 0.2, 0.8
MIN_KEEP = 200

PAIR_SOFT_EPS = 1e-6

# ---- Normalization toggles
STANDARDIZE_X_PER_FOLD = True
NORMALIZE_C_FOR_CBM = True
NORM_EPS = 1e-6

# ---- Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def task_signature(*, use_train_only=USE_TRAIN_ONLY, max_per_class=MAX_PER_CLASS, seed_data=SEED_DATA,
                   concept_names=CONCEPT_NAMES, clip_model_name=CLIP_MODEL_NAME, k_folds=K_FOLDS) -> str:
    payload = {
        "DATASET": "CUB-200-2011",
        "TASK": "Gull_vs_Tern",
        "USE_TRAIN_ONLY": bool(use_train_only),
        "MAX_PER_CLASS": max_per_class,
        "SEED_DATA": int(seed_data),
        "CONCEPTS": list(concept_names),
        "CLIP_MODEL": clip_model_name,
        "K_FOLDS": int(k_folds),
    }
    return hashlib.md5(repr(payload).encode("utf-8")).hexdigest()[:10]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def default_results_path(*, k_folds=K_FOLDS, ts=None) -> Path:
    return RESULTS_DIR / f"results_cub_gulltern.csv"
