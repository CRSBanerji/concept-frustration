"""Data loading, text embedding, and fold generation for the sarcasm task."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from .config import (
    DATA_DIR,
    FILES,
    N_DATA,
    SEED_DATA,
    SENT_MODEL,
    EMB_MODEL,
    BATCH_SIZE_SENT,
    BATCH_SIZE_EMB,
    MAX_LEN,
    device,
)


def load_jsonl(path) -> pd.DataFrame:
    """Load a JSON-lines file where each line is a JSON object."""
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _subsample_indices(n_total: int, n_keep: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_keep >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return rng.choice(np.arange(n_total, dtype=np.int64), size=n_keep, replace=False)


def _batched_roberta_sentiment_logits(texts, model_name: str, *, batch_size: int, max_len: int, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    mdl.eval()

    id2label = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
    if any(("neg" in v or "negative" in v) for v in id2label.values()):
        def _find(targets):
            for i, lab in id2label.items():
                if any(t in lab for t in targets):
                    return i
            return None
        neg_i = _find(["neg", "negative"])
        neu_i = _find(["neu", "neutral"])
        pos_i = _find(["pos", "positive"])
        if None in (neg_i, neu_i, pos_i):
            neg_i, neu_i, pos_i = 0, 1, 2
    else:
        neg_i, neu_i, pos_i = 0, 1, 2

    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = [("" if t is None else str(t)) for t in texts[i:i + batch_size]]
        inputs = tok(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len,
        ).to(device)
        logits = mdl(**inputs).logits
        all_logits.append(logits.detach().cpu())

    L = np.concatenate([x.numpy() for x in all_logits], axis=0).astype(np.float32)
    z_neg = L[:, neg_i]
    z_neu = L[:, neu_i]
    z_pos = L[:, pos_i]
    C = np.stack([z_pos, z_neg, z_neu], axis=1).astype(np.float32)
    return C


def _batched_deberta_cls_embeddings(texts, model_name: str, *, batch_size: int, max_len: int, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()

    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = [("" if t is None else str(t)) for t in texts[i:i + batch_size]]
        inputs = tok(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len,
        ).to(device)

        out = mdl(**inputs)
        cls = out.last_hidden_state[:, 0, :]
        vecs.append(cls.detach().cpu().numpy())

    X = np.concatenate(vecs, axis=0).astype(np.float32)
    return X


def build_or_load_cached_dataset():
    """
    Build dataset from raw files.
    Cache has been removed for repository simplicity and reproducibility.
    """
    frames = []
    for fn in FILES:
        path = DATA_DIR / fn
        if not path.exists():
            raise FileNotFoundError(f"Missing sarcasm data file: {path}")
        df0 = load_jsonl(path)
        frames.append(df0)

    df = pd.concat(frames, axis=0, ignore_index=True)

    if "headline" not in df.columns:
        raise ValueError(f"Expected column 'headline' in sarcasm dataset, got columns: {list(df.columns)}")
    if "is_sarcastic" not in df.columns:
        raise ValueError(f"Expected column 'is_sarcastic' in sarcasm dataset, got columns: {list(df.columns)}")

    df = df.dropna(subset=["headline", "is_sarcastic"]).copy()
    df["headline"] = df["headline"].astype(str)
    df["is_sarcastic"] = df["is_sarcastic"].astype(int)
    df = df[df["is_sarcastic"].isin([0, 1])].reset_index(drop=True)

    print(
        f"[DEBUG] Sarcasm full size={len(df)} | prevalence={df['is_sarcastic'].mean():.4f} "
        f"({df['is_sarcastic'].sum()}/{len(df)})"
    )

    idx = _subsample_indices(len(df), N_DATA, SEED_DATA)
    df = df.iloc[idx].reset_index(drop=True)

    texts = df["headline"].tolist()
    Y = df["is_sarcastic"].to_numpy(dtype=np.int64)

    print(
        f"[DEBUG] Sarcasm subsample size={len(df)} | prevalence={Y.mean():.4f} "
        f"({int(Y.sum())}/{len(Y)})"
    )

    print("Computing RoBERTa sentiment logits (concepts)...")
    C = _batched_roberta_sentiment_logits(
        texts, SENT_MODEL, batch_size=BATCH_SIZE_SENT, max_len=MAX_LEN, device=device
    )

    print("Computing DeBERTa CLS embeddings (activations a)...")
    X = _batched_deberta_cls_embeddings(
        texts, EMB_MODEL, batch_size=BATCH_SIZE_EMB, max_len=MAX_LEN, device=device
    )

    return X, C, Y, texts


def make_stratified_folds(y: np.ndarray, k_folds: int, seed: int):
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_folds = np.array_split(pos, k_folds)
    neg_folds = np.array_split(neg, k_folds)

    folds = []
    all_idx = np.arange(len(y), dtype=np.int64)
    for k in range(k_folds):
        te = np.concatenate([pos_folds[k], neg_folds[k]])
        te = np.unique(te)
        mask = np.ones(len(y), dtype=bool)
        mask[te] = False
        tr = all_idx[mask]
        folds.append((tr, te))
    return folds
