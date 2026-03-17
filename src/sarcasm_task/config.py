
"""Configuration for the sarcasm concept-frustration experiment."""

from pathlib import Path
import hashlib
import torch

SPLIT = "train"   # kept for filename compatibility; not used for local JSONL
N_DATA = 10000
SEED_DATA = 123

# Local sarcasm dataset config
DATA_DIR_NAME = "headlines_data"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = REPO_ROOT / "data" / DATA_DIR_NAME
RESULTS_DIR = REPO_ROOT / "results" / "sarcasm"

FILES = [
    "Sarcasm_Headlines_Dataset.json",
    "Sarcasm_Headlines_Dataset_v2.json",
]


K_FOLDS = 10
SEED_FOLDS = 777

BATCH_SIZE_SENT = 64
BATCH_SIZE_EMB  = 64
MAX_LEN = 128

SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
EMB_MODEL  = "microsoft/deberta-v3-base"

# Training params
P_LO, P_HI = 0.2, 0.8
MIN_KEEP = 500

HIDDEN_BB = 128
EPOCHS_BB = 60

EPOCHS_CBM = 30
LAMBDA_C = 1.0

K_SAE = 300
EPOCHS_SAE = 60

PAIR_SOFT_EPS = 1e-6

# normalization toggles
STANDARDIZE_X_PER_FOLD = True
NORMALIZE_C_FOR_CBM = True
NORM_EPS = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _norm_label(s: str) -> str:
    s = str(s).strip().lower()
    return s[6:] if s.startswith("label_") else s


def _task_signature() -> str:
    payload = {
        "DATASET": "sarcasm_headlines_jsonl",
        "DATA_DIR": str(DATA_DIR),
        "FILES": list(FILES),
        "SPLIT": SPLIT,
        "N_DATA": int(N_DATA),
        "SEED_DATA": int(SEED_DATA),
        "SENT_MODEL": SENT_MODEL,
        "EMB_MODEL": EMB_MODEL,
        "MAX_LEN": int(MAX_LEN),
    }
    b = repr(payload).encode("utf-8")
    return hashlib.md5(b).hexdigest()[:10]


TASK_SIG = _task_signature()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "results_sarcasm.csv"
