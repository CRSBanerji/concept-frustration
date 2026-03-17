from pathlib import Path
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pointbiserialr
import torch
from transformers import CLIPProcessor, CLIPModel

from .config import (
    DATA_DIR, USE_TRAIN_ONLY, MAX_PER_CLASS, SEED_DATA, CONCEPT_NAMES,
    CLIP_MODEL_NAME, BATCH_SIZE_EMB, device
)

def _autodetect_cub_paths():
    """
    Supports:
      (A) standard: ./CUB_200_2011/{images.txt, attributes/attributes.txt, ...}
      (B) nested:   ./CUB_200_2011/attributes.txt + ./CUB_200_2011/CUB_200_2011/{...}
    """
    base = str(DATA_DIR)

    std_root = base
    std_attr_names = os.path.join(std_root, "attributes", "attributes.txt")

    nested_root = os.path.join(base, "CUB_200_2011")
    nested_attr_names = os.path.join(base, "attributes.txt")

    def _exists_all(paths):
        return all(os.path.exists(p) for p in paths)

    if _exists_all([os.path.join(std_root, "images.txt"), std_attr_names]):
        return std_root, std_attr_names, os.path.join(std_root, "images")
    if _exists_all([os.path.join(nested_root, "images.txt"), nested_attr_names]):
        return nested_root, nested_attr_names, os.path.join(nested_root, "images")

    raise FileNotFoundError(
        f"Could not auto-detect CUB layout under {DATA_DIR}. "
    )

def _read_two_col(path, c1, c2):
    return pd.read_csv(path, sep=r"\s+", header=None, names=[c1, c2], engine="python")

def _get_attr_id(attr_names_df: pd.DataFrame, attr_name: str) -> int:
    row = attr_names_df[attr_names_df["attr_name"] == attr_name]
    if row.empty:
        # show partials to help debugging
        partial = attr_names_df[attr_names_df["attr_name"].str.contains(attr_name.split("::")[-1], regex=False)]
        raise ValueError(f"Attribute not found: {attr_name}\nPartial matches:\n{partial.head(25)}")
    return int(row.iloc[0]["attr_id"])  # 1-based

def _embed_images_clip(image_paths: list[str], model_name: str, *, batch_size: int, device):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(img)

        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        f = model.get_image_features(**inputs)  # [B, d]
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)  # normalize
        feats.append(f.detach().cpu())

    X = torch.cat(feats, dim=0).numpy().astype(np.float32)
    return X

def build_or_load_cached_dataset():
    CUB_ROOT, ATTR_NAMES_PATH, IMAGES_DIR = _autodetect_cub_paths()
    print("Using CUB_ROOT:", CUB_ROOT)
    print("Using ATTR_NAMES_PATH:", ATTR_NAMES_PATH)
    print("Using IMAGES_DIR:", IMAGES_DIR)

    images = _read_two_col(os.path.join(CUB_ROOT, "images.txt"), "image_id", "rel_path")
    labels = _read_two_col(os.path.join(CUB_ROOT, "image_class_labels.txt"), "image_id", "class_id")
    split  = _read_two_col(os.path.join(CUB_ROOT, "train_test_split.txt"), "image_id", "is_train")
    classes = _read_two_col(os.path.join(CUB_ROOT, "classes.txt"), "class_id", "class_name")
    attr_names = _read_two_col(ATTR_NAMES_PATH, "attr_id", "attr_name")

    gull_classes = classes[classes["class_name"].str.contains("Gull", regex=False)]
    tern_classes = classes[classes["class_name"].str.contains("Tern", regex=False)]
    gull_ids = gull_classes["class_id"].tolist()
    tern_ids = tern_classes["class_id"].tolist()

    print(f"[DEBUG] Gull species={len(gull_ids)} | Tern species={len(tern_ids)}")

    dfA = labels[labels["class_id"].isin(gull_ids)].copy()
    dfB = labels[labels["class_id"].isin(tern_ids)].copy()
    dfA["Y"] = 1
    dfB["Y"] = 0
    dfY = pd.concat([dfA, dfB], ignore_index=True)

    if USE_TRAIN_ONLY:
        dfY = dfY.merge(split, on="image_id", how="inner")
        dfY = dfY[dfY["is_train"] == 1].copy()

    if MAX_PER_CLASS is not None:
        rng = np.random.default_rng(SEED_DATA)
        kept = []
        for _, sub in dfY.groupby("class_id", sort=False):
            if len(sub) <= MAX_PER_CLASS:
                kept.append(sub)
            else:
                take = rng.choice(sub.index.to_numpy(), size=MAX_PER_CLASS, replace=False)
                kept.append(sub.loc[take])
        dfY = pd.concat(kept, ignore_index=True)

    dfY = dfY.merge(images, on="image_id", how="left")
    dfY = dfY.dropna(subset=["rel_path"]).reset_index(drop=True)

    abs_paths = [os.path.join(IMAGES_DIR, rp) for rp in dfY["rel_path"].tolist()]
    missing = [p for p in abs_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image files. Example: {missing[0]}")

    Y = dfY["Y"].to_numpy(dtype=np.int64)
    image_ids = dfY["image_id"].to_numpy(dtype=np.int64)
    rel_paths = dfY["rel_path"].tolist()

    print(
        f"[DEBUG] N={len(Y)} | pos_rate(gull)={Y.mean():.3f} | "
        f"gull_images={int(Y.sum())} tern_images={int((1-Y).sum())}"
    )

    attr_ids = [_get_attr_id(attr_names, nm) for nm in CONCEPT_NAMES]
    print("[DEBUG] Concept attr_ids (1-based):", list(zip(CONCEPT_NAMES, attr_ids)))

    img_attr_path = os.path.join(CUB_ROOT, "attributes", "image_attribute_labels.txt")
    keep_ids = set(image_ids.tolist())
    wanted_attr_ids = set(attr_ids)

    rows = []
    for chunk in pd.read_csv(
        img_attr_path,
        sep=r"\s+",
        header=None,
        names=["image_id", "attr_id", "is_present", "certainty", "time"],
        chunksize=1_000_000,
        engine="c",
        on_bad_lines="skip",
    ):
        chunk = chunk[chunk["image_id"].isin(keep_ids)]
        chunk = chunk[chunk["attr_id"].isin(wanted_attr_ids)]
        if not chunk.empty:
            rows.append(chunk)

    if not rows:
        raise RuntimeError("No attribute rows found for selected images/attributes (check paths).")

    df_attr = pd.concat(rows, ignore_index=True)

    C_wide = df_attr.pivot_table(
        index="image_id",
        columns="attr_id",
        values="is_present",
        aggfunc="max"
    ).fillna(0.0)

    for aid in attr_ids:
        if aid not in C_wide.columns:
            C_wide[aid] = 0.0
    C_wide = C_wide[attr_ids]
    C_wide = C_wide.reindex(image_ids).fillna(0.0)

    C = C_wide.to_numpy(dtype=np.float32)

    print("\n[DEBUG] Concept ↔ Y (point-biserial):")
    for j, nm in enumerate(CONCEPT_NAMES):
        r, p = pointbiserialr(Y.astype(float), C[:, j].astype(float))
        print(f"  {nm:35s} r={r:+.3f} p={p:.2e}")

    print("\nComputing CLIP embeddings...")
    X = _embed_images_clip(abs_paths, CLIP_MODEL_NAME, batch_size=BATCH_SIZE_EMB, device=device)
    print("X shape:", X.shape, "C shape:", C.shape, "Y shape:", Y.shape)

    return X, C, Y, image_ids, rel_paths

def make_stratified_folds(y: np.ndarray, k_folds: int, seed: int):
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(pos); rng.shuffle(neg)

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
