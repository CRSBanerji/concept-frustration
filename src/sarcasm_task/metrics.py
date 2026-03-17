"""Metrics and geometry utilities for the sarcasm task."""

import numpy as np
import torch

def accuracy_from_logits(logits: torch.Tensor, y_true_np: np.ndarray) -> float:
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)
    return float((preds == y_true_np).mean())

def f1_from_logits(logits: torch.Tensor, y_true_np: np.ndarray, thresh: float = 0.5) -> float:
    y_true = np.asarray(y_true_np, dtype=np.int64).reshape(-1)
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y_pred = (probs >= thresh).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0

def fisher_cosine_matrix(W: np.ndarray, D: np.ndarray, Fm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Fsym = 0.5 * (Fm + Fm.T)
    WFW = np.einsum("ih,hk,ik->i", W, Fsym, W)
    DFD = np.einsum("jh,hk,jk->j", D, Fsym, D)
    Wn = np.sqrt(np.maximum(WFW, eps))
    Dn = np.sqrt(np.maximum(DFD, eps))
    num = W @ Fsym @ D.T
    return num / (Wn[:, None] * Dn[None, :] + eps)

def euclid_cosine_matrix(W: np.ndarray, D: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Wn = np.sqrt(np.sum(W * W, axis=1, keepdims=True) + eps)
    Dn = np.sqrt(np.sum(D * D, axis=1, keepdims=True) + eps)
    return (W @ D.T) / (Wn @ Dn.T)

def fisher_cosine_self(W: np.ndarray, Fm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Fsym = 0.5 * (Fm + Fm.T)
    WFW = np.einsum("ih,hk,ik->i", W, Fsym, W)
    Wn = np.sqrt(np.maximum(WFW, eps))
    num = W @ Fsym @ W.T
    return num / (Wn[:, None] * Wn[None, :] + eps)

def euclid_cosine_self(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Wn = np.sqrt(np.sum(W * W, axis=1) + eps)
    num = W @ W.T
    return num / (Wn[:, None] * Wn[None, :] + eps)

def pair_soft_frustration_metric(S: np.ndarray, Z: np.ndarray, *, eps: float = 1e-6) -> dict:
    k, K = S.shape
    if k < 2:
        return {
            "pair_soft_mean": 0.0, "pair_soft_max": 0.0, "pair_soft_nonzero_frac": 0.0,
            "pair_soft_pairs": 0, "pair_soft_eps": float(eps), "mean_abs_Z": 0.0,
            "pair_raw_mean": 0.0, "pair_raw_max": 0.0, "pair_raw_nonzero_frac": 0.0,
        }

    soft_scores, raw_scores = [], []
    soft_nonzero = 0
    raw_nonzero = 0
    total_pairs = 0

    for l in range(k):
        for r in range(l + 1, k):
            total_pairs += 1
            z = float(Z[l, r])
            if z == 0.0:
                soft_scores.append(0.0); raw_scores.append(0.0)
                continue

            prod = S[l, :] * S[r, :]
            zsign = np.sign(z)
            psign = np.sign(prod)

            contr_mask = (psign != 0) & (psign != zsign)
            if not np.any(contr_mask):
                soft_scores.append(0.0); raw_scores.append(0.0)
                continue

            best_raw = float(np.max(np.abs(prod[contr_mask])))
            raw_scores.append(best_raw)
            if best_raw > 0.0:
                raw_nonzero += 1

            best_soft = best_raw / (abs(z) + eps)
            soft_scores.append(best_soft)
            if best_soft > 0.0:
                soft_nonzero += 1

    soft_scores = np.asarray(soft_scores, dtype=float)
    raw_scores = np.asarray(raw_scores, dtype=float)

    tri = np.triu_indices(k, 1)
    mean_abs_Z = float(np.mean(np.abs(Z[tri])))

    return {
        "pair_soft_mean": float(soft_scores.mean()) if soft_scores.size else 0.0,
        "pair_soft_max": float(soft_scores.max()) if soft_scores.size else 0.0,
        "pair_soft_nonzero_frac": float(soft_nonzero / total_pairs) if total_pairs else 0.0,
        "pair_soft_pairs": int(total_pairs),
        "pair_soft_eps": float(eps),
        "mean_abs_Z": float(mean_abs_Z),
        "pair_raw_mean": float(raw_scores.mean()) if raw_scores.size else 0.0,
        "pair_raw_max": float(raw_scores.max()) if raw_scores.size else 0.0,
        "pair_raw_nonzero_frac": float(raw_nonzero / total_pairs) if total_pairs else 0.0,
    }

def frob_norm(A: np.ndarray) -> float:
    return float(np.sqrt(np.sum(A * A)))

def frob_abs_rel(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> dict:
    diff = A - B
    abs_f = frob_norm(diff)
    denom = frob_norm(A) + eps
    rel_f = abs_f / denom
    return {"frob_abs": float(abs_f), "frob_rel": float(rel_f)}

def cov_matrix(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False, bias=False)

def corr_from_cov(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(C), eps))
    return C / (d[:, None] * d[None, :] + eps)

def _select_frustrated_atoms_pair12(S: np.ndarray, Z: np.ndarray, *, tiny: float = 1e-15) -> np.ndarray:
    z = float(Z[0, 1])
    if z == 0.0:
        return np.array([], dtype=np.int64)

    prod = S[0, :] * S[1, :]
    mask_nz = np.abs(prod) > tiny
    zsign = np.sign(z)
    psign = np.sign(prod)

    contr = mask_nz & (psign != 0) & (psign != zsign)
    return np.where(contr)[0].astype(np.int64)

def _project_X_onto_atoms(X: np.ndarray, D: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    return (X @ D[idx, :].T).astype(np.float32)

def _ridge_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, *, lam: float = 1e-3):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64).reshape(-1)

    x_mu = Xtr.mean(axis=0, keepdims=True) if Xtr.shape[1] else np.zeros((1, 0), dtype=np.float64)
    y_mu = float(ytr.mean())

    Xc = Xtr - x_mu
    yc = ytr - y_mu

    d = Xc.shape[1]
    if d == 0:
        yhat = np.full((Xte.shape[0],), y_mu, dtype=np.float32)
        return yhat, (y_mu, np.zeros((0,), dtype=np.float32))

    A = Xc.T @ Xc + lam * np.eye(d)
    b = np.linalg.solve(A, Xc.T @ yc)
    b0 = y_mu - float((x_mu @ b.reshape(-1, 1)).squeeze())
    yhat = (Xte @ b) + b0
    return yhat.astype(np.float32), (float(b0), b.astype(np.float32))

def _mse_r2(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true))
    r2 = float(1.0 - mse / (var + 1e-12))
    return mse, r2
