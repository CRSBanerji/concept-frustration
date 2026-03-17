import numpy as np
import torch

def compute_fisher_on_input_x(bb: BlackBoxMLP, X: np.ndarray, *, device="cpu", ridge=1e-6):
    """
    Empirical Fisher in x-space:
      F ≈ (1/N) Σ p(1-p) g g^T
    where g(x)=∇_x logit = W_H^T (m(x) ⊙ w_l), m_i=1{(W_H x + b_H)_i>0}
    """
    bb.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)

    logits, z, _ = bb(Xt, return_hidden=True)   # z = W_H x + b_H
    p = torch.sigmoid(logits)                   # (N,)
    s = p * (1 - p)                             # (N,)
    m = (z > 0).float()                         # (N,H)

    W_H = bb.fc1.weight                         # (H,r)
    w_l = bb.fc2.weight.squeeze(0)              # (H,)

    U = m * w_l.unsqueeze(0)                    # (N,H)
    G = U @ W_H                                 # (N,r)

    F = (G.T @ (G * s.unsqueeze(1))) / float(X.shape[0])   # (r,r)
    F = F + ridge * torch.eye(F.shape[0], device=device)
    return F.cpu().numpy()

def fisher_cosine_matrix(W: np.ndarray, D: np.ndarray, F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Fsym = 0.5 * (F + F.T)
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

def fisher_cosine_self(W: np.ndarray, F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Fsym = 0.5 * (F + F.T)
    WFW = np.einsum("ih,hk,ik->i", W, Fsym, W)
    Wn = np.sqrt(np.maximum(WFW, eps))
    num = W @ Fsym @ W.T
    return num / (Wn[:, None] * Wn[None, :] + eps)

def euclid_cosine_self(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Wn = np.sqrt(np.sum(W * W, axis=1) + eps)
    num = W @ W.T
    return num / (Wn[:, None] * Wn[None, :] + eps)

def pair_raw_frustration_mean(S: np.ndarray, Z: np.ndarray) -> float:
    k, K = S.shape
    if k < 2:
        return 0.0

    scores = []
    for l in range(k):
        for r in range(l + 1, k):
            z = float(Z[l, r])
            if z == 0.0:
                scores.append(0.0)
                continue
            prod = S[l, :] * S[r, :]
            zsign = np.sign(z)
            psign = np.sign(prod)
            contr = (psign != 0) & (psign != zsign)
            if not np.any(contr):
                scores.append(0.0)
            else:
                scores.append(float(np.max(np.abs(prod[contr]))))
    return float(np.mean(scores)) if scores else 0.0

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

def compute_T_terms(
    *,
    B_known: np.ndarray,
    B_temp: np.ndarray,
    B_full: np.ndarray,
    w_star: np.ndarray,
    k_known: int,
    eps: float = 1e-12,
) -> dict:
    k_total = B_full.shape[0]
    k_unk = k_total - k_known
    if k_unk <= 0:
        return {"T1": 0.0, "T2": 0.0, "T3": 0.0, "T4": 0.0}

    psi_k = np.asarray(w_star[:k_known], dtype=np.float64)
    psi_u = np.asarray(w_star[k_known:], dtype=np.float64)

    Bkk = np.asarray(B_known, dtype=np.float64)
    Bku = np.asarray(B_full[:k_known, k_known:], dtype=np.float64)
    Buk = Bku.T
    Btemp = np.asarray(B_temp, dtype=np.float64)

    Binv = np.linalg.inv(Bkk + eps * np.eye(k_known))

    T1 = float(psi_k.T @ Bkk @ psi_k)
    T2 = float(psi_k.T @ Bku @ psi_u)

    bridge = Buk @ Binv @ Bku
    T3 = float(psi_u.T @ bridge @ psi_u)

    T4 = float(psi_u.T @ Btemp @ psi_u)

    return {"T1": T1, "T2": T2, "T3": T3, "T4": T4}
