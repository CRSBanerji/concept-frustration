"""Top-level experiment runner for the sarcasm task."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (
    device as DEFAULT_DEVICE,
    N_DATA,
    K_FOLDS,
    SEED_FOLDS,
    RESULTS_CSV,
    RESULTS_DIR,
    STANDARDIZE_X_PER_FOLD,
    NORMALIZE_C_FOR_CBM,
    NORM_EPS,
    HIDDEN_BB,
    EPOCHS_BB,
    EPOCHS_CBM,
    LAMBDA_C,
    K_SAE,
    EPOCHS_SAE,
    P_LO,
    P_HI,
    MIN_KEEP,
    PAIR_SOFT_EPS,
)
from .data import build_or_load_cached_dataset, make_stratified_folds
from .training import (
    train_bb_minibatch,
    train_cbm_linear_minibatch,
    train_sae_minibatch,
    bb_predict_proba,
    compute_fisher_on_input_x,
)
from .metrics import (
    f1_from_logits,
    fisher_cosine_matrix,
    euclid_cosine_matrix,
    fisher_cosine_self,
    euclid_cosine_self,
    pair_soft_frustration_metric,
    frob_abs_rel,
    cov_matrix,
    corr_from_cov,
)


def run_one_fold(
    *,
    fold_id: int,
    seed: int,
    X: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    device="cpu",
    p_lo=0.2,
    p_hi=0.8,
    min_keep=200,
    hidden_bb=128,
    epochs_bb=60,
    epochs_cbm=30,
    lambda_c=1.0,
    K_sae=60,
    epochs_sae=60,
    standardize_x_per_fold=True,
    normalize_c_for_cbm=True,
    norm_eps=1e-6,
    pair_soft_eps=1e-6,
):
    r = X.shape[1]
    B = np.cov(C, rowvar=False, bias=False).astype(np.float32)

    X_tr, X_te = X[tr_idx].astype(np.float32), X[te_idx].astype(np.float32)
    y_tr, y_te = y[tr_idx], y[te_idx]

    if standardize_x_per_fold:
        x_mu = X_tr.mean(axis=0, keepdims=True)
        x_sd = X_tr.std(axis=0, keepdims=True) + norm_eps
        X_tr = (X_tr - x_mu) / x_sd
        X_te = (X_te - x_mu) / x_sd

    bb, bb_acc = train_bb_minibatch(
        X_tr, y_tr, X_te, y_te,
        hidden=hidden_bb, epochs=epochs_bb, batch_size=512, lr=1e-3,
        seed=seed, device=device
    )

    with torch.no_grad():
        bb_logits_te = bb(torch.tensor(X_te, dtype=torch.float32, device=device))
        bb_f1 = f1_from_logits(bb_logits_te, y_te, thresh=0.5)

    p_lo_use, p_hi_use = (p_lo, p_hi) if p_lo < p_hi else (p_hi, p_lo)
    p_tr = bb_predict_proba(bb, X_tr, device=device)
    keep = np.where((p_tr >= p_lo_use) & (p_tr <= p_hi_use))[0]
    if keep.size < min_keep:
        order = np.argsort(np.abs(p_tr - 0.5))
        keep = order[:min_keep]
    Fm = compute_fisher_on_input_x(bb, X_tr[keep], device=device, ridge=1e-6)

    sae = train_sae_minibatch(
        X_tr,
        K=K_sae, epochs=epochs_sae, batch_size=512, lr=2e-3, l1=1e-3,
        seed=seed, device=device
    )
    D = sae.D.detach().cpu().numpy().astype(np.float32)

    C_tr_raw = C[tr_idx].astype(np.float32)
    C_te_raw = C[te_idx].astype(np.float32)

    if normalize_c_for_cbm:
        c_mu = C_tr_raw.mean(axis=0, keepdims=True)
        c_sd = C_tr_raw.std(axis=0, keepdims=True) + norm_eps
        C_tr_norm = (C_tr_raw - c_mu) / c_sd
        C_te_norm = (C_te_raw - c_mu) / c_sd
    else:
        c_mu = np.zeros((1, C.shape[1]), dtype=np.float32)
        c_sd = np.ones((1, C.shape[1]), dtype=np.float32)
        C_tr_norm, C_te_norm = C_tr_raw, C_te_raw

    def _run_cbm_block(k_known: int, tag: str):
        Ck_tr = C_tr_norm[:, :k_known].astype(np.float32)
        Ck_te = C_te_norm[:, :k_known].astype(np.float32)
        B_known = np.array(B[:k_known, :k_known], dtype=float)

        cbm, cbm_acc, cbm_mse = train_cbm_linear_minibatch(
            X_tr, Ck_tr, y_tr,
            X_te, Ck_te, y_te,
            epochs=epochs_cbm, batch_size=512, lr=1e-3,
            lambda_c=lambda_c, seed=seed, device=device
        )
        Wc = cbm.concept.weight.detach().cpu().numpy().astype(np.float32)

        with torch.no_grad():
            Xv = torch.tensor(X_te, dtype=torch.float32, device=device)
            c_hat_te, logit_te = cbm(Xv)
            C_hat_te_norm = c_hat_te.detach().cpu().numpy().astype(np.float32)
            cbm_f1 = f1_from_logits(logit_te, y_te, thresh=0.5)

        if normalize_c_for_cbm:
            mu_k = c_mu[:, :k_known]
            sd_k = c_sd[:, :k_known]
            C_hat_te_raw = (C_hat_te_norm * sd_k) + mu_k
        else:
            C_hat_te_raw = C_hat_te_norm

        S_f = fisher_cosine_matrix(Wc, D, Fm)
        S_e = euclid_cosine_matrix(Wc, D)
        Z_f = fisher_cosine_self(Wc, Fm)
        Z_e = euclid_cosine_self(Wc)

        pairF = pair_soft_frustration_metric(S_f, Z_f, eps=pair_soft_eps)
        pairE = pair_soft_frustration_metric(S_e, Z_e, eps=pair_soft_eps)

        Sd = frob_abs_rel(S_f, S_e)

        Cov_hat = cov_matrix(C_hat_te_raw)
        Corr_hat = corr_from_cov(Cov_hat)
        Corr_B = corr_from_cov(B_known)
        cov_diff = frob_abs_rel(Cov_hat, B_known)
        corr_diff = frob_abs_rel(Corr_hat, Corr_B)

        out = {
            f"{tag}_k_known": int(k_known),
            f"{tag}_acc": float(cbm_acc),
            f"{tag}_f1": float(cbm_f1),
            f"{tag}_mse": float(cbm_mse),
            f"{tag}_F_pair_raw_mean": float(pairF["pair_raw_mean"]),
            f"{tag}_E_pair_raw_mean": float(pairE["pair_raw_mean"]),
            f"{tag}_F_pair_raw_max": float(pairF["pair_raw_max"]),
            f"{tag}_E_pair_raw_max": float(pairE["pair_raw_max"]),
            f"{tag}_S_frob_abs": float(Sd["frob_abs"]),
            f"{tag}_S_frob_rel": float(Sd["frob_rel"]),
            f"{tag}_Cov_frob_abs": float(cov_diff["frob_abs"]),
            f"{tag}_Cov_frob_rel": float(cov_diff["frob_rel"]),
            f"{tag}_Corr_frob_abs": float(corr_diff["frob_abs"]),
            f"{tag}_Corr_frob_rel": float(corr_diff["frob_rel"]),
        }
        return out, (S_f, Z_f, Wc)

    cbm1_metrics, (_S1_f, _Z1_f, _W1) = _run_cbm_block(2, "cbm1")
    cbm2_metrics, (_S2_f, _Z2_f, W2) = _run_cbm_block(3, "cbm2")

    W2_12 = W2[:2, :]
    S2_f_12 = fisher_cosine_matrix(W2_12, D, Fm)
    S2_e_12 = euclid_cosine_matrix(W2_12, D)
    Z2_f_12 = fisher_cosine_self(W2_12, Fm)
    Z2_e_12 = euclid_cosine_self(W2_12)

    pairF2_12 = pair_soft_frustration_metric(S2_f_12, Z2_f_12, eps=pair_soft_eps)
    pairE2_12 = pair_soft_frustration_metric(S2_e_12, Z2_e_12, eps=pair_soft_eps)

    cbm2_pair12_metrics = {
        "cbm2_F_pair12_raw_mean": float(pairF2_12["pair_raw_mean"]),
        "cbm2_E_pair12_raw_mean": float(pairE2_12["pair_raw_mean"]),
        "cbm2_F_pair12_raw_max": float(pairF2_12["pair_raw_max"]),
        "cbm2_E_pair12_raw_max": float(pairE2_12["pair_raw_max"]),
        "cbm2_F_pair12_nonzero_frac": float(pairF2_12["pair_raw_nonzero_frac"]),
        "cbm2_E_pair12_nonzero_frac": float(pairE2_12["pair_raw_nonzero_frac"]),
    }

    row = {
        "stdX_per_fold": int(standardize_x_per_fold),
        "normC_for_cbm": int(normalize_c_for_cbm),
        "fold": int(fold_id),
        "seed": int(seed),
        "N_total": int(len(X)),
        "N_train": int(len(tr_idx)),
        "N_test": int(len(te_idx)),
        "r": int(r),
        "p_lo": float(p_lo_use),
        "p_hi": float(p_hi_use),
        "F_keep_n": int(len(keep)),
        "bb_acc": float(bb_acc),
        "bb_f1": float(bb_f1),
        "Y_pos_rate_total": float(np.mean(y)),
        "Y_pos_rate_train": float(np.mean(y_tr)),
        "Y_pos_rate_test": float(np.mean(y_te)),
    }
    row.update(cbm1_metrics)
    row.update(cbm2_metrics)
    row.update(cbm2_pair12_metrics)
    return row


def run_experiment(
    *,
    X=None,
    C=None,
    Y=None,
    n_data=N_DATA,
    k_folds=K_FOLDS,
    seed_data=None,
    seed_folds=SEED_FOLDS,
    device=None,
    p_lo=P_LO,
    p_hi=P_HI,
    min_keep=MIN_KEEP,
    hidden_bb=HIDDEN_BB,
    epochs_bb=EPOCHS_BB,
    epochs_cbm=EPOCHS_CBM,
    lambda_c=LAMBDA_C,
    K_sae=K_SAE,
    epochs_sae=EPOCHS_SAE,
    standardize_x_per_fold=STANDARDIZE_X_PER_FOLD,
    normalize_c_for_cbm=NORMALIZE_C_FOR_CBM,
    norm_eps=NORM_EPS,
    pair_soft_eps=PAIR_SOFT_EPS,
):
    if device is None:
        device = str(DEFAULT_DEVICE)
    else:
        device = str(device)

    if X is None or C is None or Y is None:
        X, C, Y, _texts = build_or_load_cached_dataset()

    if n_data is not None and n_data < len(Y):
        rng = np.random.default_rng(SEED_DATA if seed_data is None else seed_data)
        idx = rng.choice(np.arange(len(Y)), size=n_data, replace=False)
        X = X[idx]
        C = C[idx]
        Y = Y[idx]

    folds = make_stratified_folds(Y, k_folds, seed_folds)

    rows = []
    for k, (tr_idx, te_idx) in enumerate(folds):
        print(f"\n=== Fold {k + 1}/{k_folds} ===")
        out = run_one_fold(
            fold_id=k,
            seed=seed_folds + k,
            X=X, C=C, y=Y,
            tr_idx=tr_idx, te_idx=te_idx,
            device=device,
            p_lo=p_lo, p_hi=p_hi, min_keep=min_keep,
            hidden_bb=hidden_bb,
            epochs_bb=epochs_bb,
            epochs_cbm=epochs_cbm,
            lambda_c=lambda_c,
            K_sae=K_sae,
            epochs_sae=epochs_sae,
            standardize_x_per_fold=standardize_x_per_fold,
            normalize_c_for_cbm=normalize_c_for_cbm,
            norm_eps=norm_eps,
            pair_soft_eps=pair_soft_eps,
        )
        rows.append(out)

        print(
            f"bb_acc={out['bb_acc']:.3f} bb_f1={out['bb_f1']:.3f} | "
            f"cbm1_acc={out['cbm1_acc']:.3f} cbm1_f1={out['cbm1_f1']:.3f} "
            f"GammaF1={out['cbm1_F_pair_raw_mean']:.4f} | "
            f"cbm2_acc={out['cbm2_acc']:.3f} cbm2_f1={out['cbm2_f1']:.3f} "
            f"GammaF2(allpairs)={out['cbm2_F_pair_raw_mean']:.4f} "
            f"GammaF2(pair12)={out['cbm2_F_pair12_raw_mean']:.4f}"
        )

    return rows


def save_results(rows, out_path=RESULTS_CSV):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved results: {out_path}")
    return df


def summarize_results(rows):
    df = pd.DataFrame(rows)
    summary = df.describe(include="all").T
    cols = [c for c in ["mean", "std", "min", "max"] if c in summary.columns]
    return df, summary[cols]


def main():
    print(f"\n=== Running SARCASM HEADLINES | N_DATA={N_DATA} ===")
    print(
        f"    STANDARDIZE_X_PER_FOLD={STANDARDIZE_X_PER_FOLD} | "
        f"NORMALIZE_C_FOR_CBM={NORMALIZE_C_FOR_CBM}"
    )
    rows = run_experiment(
        n_data=N_DATA,
        k_folds=K_FOLDS,
        seed_folds=SEED_FOLDS,
        device=str(DEFAULT_DEVICE),
        p_lo=P_LO,
        p_hi=P_HI,
        min_keep=MIN_KEEP,
        hidden_bb=HIDDEN_BB,
        epochs_bb=EPOCHS_BB,
        epochs_cbm=EPOCHS_CBM,
        lambda_c=LAMBDA_C,
        K_sae=K_SAE,
        epochs_sae=EPOCHS_SAE,
        standardize_x_per_fold=STANDARDIZE_X_PER_FOLD,
        normalize_c_for_cbm=NORMALIZE_C_FOR_CBM,
        norm_eps=NORM_EPS,
        pair_soft_eps=PAIR_SOFT_EPS,
    )
    df = save_results(rows, RESULTS_CSV)
    print(summarize_results(rows)[1])
    return df
