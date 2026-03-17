import json
import numpy as np
import torch

from .data import sample_B_components_for_seed, generate_toy_dataset_concepts_first
from .training import (
    accuracy_from_logits,
    train_test_split_indices,
    bb_predict_proba,
    train_bb_minibatch,
    train_sae_minibatch,
)
from .metrics import (
    compute_fisher_on_input_x,
    fisher_cosine_matrix,
    euclid_cosine_matrix,
    fisher_cosine_self,
    euclid_cosine_self,
    pair_raw_frustration_mean,
    frob_norm,
    frob_abs_rel,
    cov_matrix,
    compute_T_terms,
)

from .training import train_cbm_hard_two_stage_ground_truth as train_cbm_hard_two_stage

def run_one(
    *,
    alpha: float,
    omega: float,
    seed: int,
    k_known: int,
    B_components,
    device="cpu",
    sigma_x: float = 0.3,
    sigma_y: float = 1.5,
    p_lo: float = 0.4,
    p_hi: float = 0.6,
    min_keep: int = 50,
    r: int = 64,
    k_total: int = 50,
    n: int = 8000,
    K_sae: int = 60,
):
    # Generate data 
    X, C, y, B, A, w, w_star, comps_out = generate_toy_dataset_concepts_first(
        n=int(n), r=int(r), k=int(k_total), k_known=int(k_known),
        sigma_x=float(sigma_x),
        sigma_y=float(sigma_y),
        omega=float(omega),
        seed=int(seed),
        alpha=float(alpha),
        A_scale="none",
        B_components=B_components,
    )

    B = np.asarray(B, dtype=np.float64)
    w_star = np.asarray(w_star, dtype=np.float64)

    B_known = np.asarray(B_components[0], dtype=np.float64)
    B_temp  = np.asarray(B_components[1], dtype=np.float64)

    rng = np.random.default_rng(seed)
    tr_idx, te_idx = train_test_split_indices(len(X), 0.75, rng)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    Ck_tr, Ck_te = C[tr_idx, :k_known], C[te_idx, :k_known]

    # (1) BB
    bb, bb_acc = train_bb_minibatch(
        X_tr, y_tr, X_te, y_te,
        hidden=128, epochs=30, batch_size=512, lr=1e-3,
        seed=seed, device=device
    )

    # (2) Fisher on uncertain subset
    p_lo_use, p_hi_use = (p_lo, p_hi) if p_lo < p_hi else (p_hi, p_lo)
    p_tr = bb_predict_proba(bb, X_tr, device=device)
    keep = np.where((p_tr >= p_lo_use) & (p_tr <= p_hi_use))[0]
    if keep.size < min_keep:
        order = np.argsort(np.abs(p_tr - 0.5))
        keep = order[:min_keep]
    F = compute_fisher_on_input_x(bb, X_tr[keep], device=device, ridge=1e-6)

    # (3) HARD CBM (2-stage: concepts -> freeze -> task)
    cbm, cbm_acc, cbm_mse = train_cbm_hard_two_stage(
        X_tr, Ck_tr, y_tr,
        X_te, Ck_te, y_te,
        concept_epochs=30,
        task_epochs=30,
        batch_size=512,
        lr_concept=1e-3,
        lr_task=1e-3,
        seed=seed,
        device=device
    )
    Wc = cbm.concept.weight.detach().cpu().numpy()

    with torch.no_grad():
        Xv = torch.tensor(X_te, dtype=torch.float32, device=device)
        c_hat_te = cbm.concept(Xv)     # concept head only
        C_hat_te = c_hat_te.detach().cpu().numpy()

    # (4) SAE
    sae = train_sae_minibatch(
        X_tr,
        K=K_sae, epochs=60, batch_size=512, lr=2e-3, l1=1e-3,
        seed=seed, device=device
    )
    D = sae.D.detach().cpu().numpy()

    # (5) S + Z matrices
    S_f = fisher_cosine_matrix(Wc, D, F)
    S_e = euclid_cosine_matrix(Wc, D)

    Z_f = fisher_cosine_self(Wc, F)
    Z_e = euclid_cosine_self(Wc)

    # (6) gamma metrics
    gamma_F = pair_raw_frustration_mean(S_f, Z_f)
    gamma_E = pair_raw_frustration_mean(S_e, Z_e)

    # (7) geometry difference (G)
    Sd = frob_abs_rel(S_f, S_e)

    # (8) covariance faithfulness (beta)
    Cov_hat = cov_matrix(C_hat_te)
    cov_diff = frob_abs_rel(Cov_hat, B_known)

    # (9) T terms
    T = compute_T_terms(
        B_known=B_known,
        B_temp=B_temp,
        B_full=B,
        w_star=w_star,
        k_known=k_known,
    )

    return {
        "seed": int(seed),
        "k_known": int(k_known),
        "k_total": int(k_total),
        "alpha": float(alpha),
        "omega": float(omega),

        "n": int(n),
        "r": int(r),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),

        "p_lo": float(p_lo_use),
        "p_hi": float(p_hi_use),
        "F_keep_n": int(len(keep)),

        "bb_acc": float(bb_acc),
        "cbm_acc": float(cbm_acc),
        "cbm_mse": float(cbm_mse),

        "F_pair_raw_mean": float(gamma_F),
        "E_pair_raw_mean": float(gamma_E),

        "S_frob_abs": float(Sd["frob_abs"]),
        "S_frob_rel": float(Sd["frob_rel"]),

        "Cov_frob_abs": float(cov_diff["frob_abs"]),
        "Cov_frob_rel": float(cov_diff["frob_rel"]),

        "T1": float(T["T1"]),
        "T2": float(T["T2"]),
        "T3": float(T["T3"]),
        "T4": float(T["T4"]),
    }

def run_sweep(
    *,
    omega_list,
    k_known_list,
    seeds,
    device="cpu",
    sigma_x=0.3,
    sigma_y=0.6,
    p_lo=0.2,
    p_hi=0.8,
    min_keep=200,
    r=64,
    k_total=50,
    n=8000,
    K_sae=60,
    alpha_strength=1.0,
):
    rows = []
    alphas = (-1.0, 0.0, +1.0)

    total = len(seeds) * len(k_known_list) * len(omega_list) * len(alphas)
    t = 0

    for seed in seeds:
        for k_known in k_known_list:
            B_components = sample_B_components_for_seed(
                k=int(k_total),
                k_known=int(k_known),
                seed=int(seed),
                alpha_strength=float(alpha_strength),
            )

            for omega in omega_list:
                for alpha in alphas:
                    t += 1
                    out = run_one(
                        alpha=alpha,
                        omega=omega,
                        seed=seed,
                        k_known=k_known,
                        B_components=B_components,
                        device=device,
                        sigma_x=sigma_x,
                        sigma_y=sigma_y,
                        p_lo=p_lo,
                        p_hi=p_hi,
                        min_keep=min_keep,
                        r=r,
                        k_total=k_total,
                        n=n,
                        K_sae=K_sae,
                    )
                    rows.append(out)

                    print(
                        f"[{t:04d}/{total:04d}] "
                        f"seed={out['seed']:2d} k={out['k_known']:2d} "
                        f"α={out['alpha']:+.0f} ω={out['omega']:.2f} keep={out['F_keep_n']:4d} | "
                        f"bb={out['bb_acc']:.3f} cbm={out['cbm_acc']:.3f} mse={out['cbm_mse']:.3f} | "
                        f"γF={out['F_pair_raw_mean']:.3f} γE={out['E_pair_raw_mean']:.3f} | "
                        f"Grel={out['S_frob_rel']:.3f} βrel={out['Cov_frob_rel']:.3f} | "
                        f"T1={out['T1']:.2e} T2={out['T2']:.2e} T3={out['T3']:.2e} T4={out['T4']:.2e}"
                    )

    return rows
