"""Configuration for the sarcasm concept-frustration experiment."""

from pathlib import Path
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = REPO_ROOT / "data" / "headlines_data"
RESULTS_DIR = REPO_ROOT / "results" / "sarcasm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "Sarcasm_Headlines_Dataset.json",
    "Sarcasm_Headlines_Dataset_v2.json",
]

N_DATA = 10000
SEED_DATA = 123

K_FOLDS = 10
SEED_FOLDS = 777

BATCH_SIZE_SENT = 64
BATCH_SIZE_EMB = 64
MAX_LEN = 128

SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
EMB_MODEL = "microsoft/deberta-v3-base"

P_LO, P_HI = 0.2, 0.8
MIN_KEEP = 500

HIDDEN_BB = 128
EPOCHS_BB = 60

EPOCHS_CBM = 30
LAMBDA_C = 1.0

K_SAE = 300
EPOCHS_SAE = 60

PAIR_SOFT_EPS = 1e-6

STANDARDIZE_X_PER_FOLD = True
NORMALIZE_C_FOR_CBM = True
NORM_EPS = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_CSV = RESULTS_DIR / "results_sarcasm.csv"
