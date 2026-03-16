import numpy as np

import torch



from .data import generate_geo_dataset, train_test_split_indices

from .training import bb_predict_proba, compute_fisher_on_input_x, train_bb_minibatch, train_cbm_linear_minibatch, train_sae_minibatch

from .metrics import (fisher_cosine_matrix, euclid_cosine_matrix, fisher_cosine_self, euclid_cosine_self, pair_soft_frustration_metric, metrics_from_S_trimmed, _select_frustrated_atoms_pair12, _project_X_onto_atoms, _ridge_predict, _mse_r2)



def run_one(
    *,
    nu: int,            # scenario id (0=cylinder, 1=sphere)
    alpha: float,       # UNUSED (compat)
    omega: float,       # UNUSED (compat)
    seed: int,
    k_known: int,       # ignored; forced to 2
    device="cpu",
    sigma_x: float = 0.3,
    p_lo: float = 0.2,
    p_hi: float = 0.8,
    min_keep: int = 200,
    r: int = 64,
    n: int = 8000,
    K_sae: int = 60,
    pair_soft_eps: float = 1e-6,
    d_max: float = 1.0,    # ### CHANGED ### default now 1.0 (depth in [0,1])
    R_task: float = 0.75,
    ridge_lam_c3: float = 1e-3,   # NEW
):
    # Force concept dimensionality: (c1,c2 known; c3 unknown)
    k_total = 3
    k_known = 2

    X, C, y, B = generate_geo_dataset(
        scenario=int(nu),
        n=int(n),
        r=int(r),
        sigma_x=float(sigma_x),
        seed=int(seed),
        d_max=float(d_max),
        R_task=float(R_task),
    )

    B_known = np.array(B[:k_known, :k_known], dtype=float)

    rng = np.random.default_rng(seed)
    tr_idx, te_idx = train_test_split_indices(len(X), 0.75, rng)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    Ck_tr, Ck_te = C[tr_idx, :k_known], C[te_idx, :k_known]

    # True unknown concept
    C3_tr = C[tr_idx, 2].astype(np.float32)
    C3_te = C[te_idx, 2].astype(np.float32)

    # (1) Black-box
    bb, bb_acc = train_bb_minibatch(
        X_tr, y_tr, X_te, y_te,
        hidden=128, epochs=30, batch_size=512, lr=1e-3,
        seed=seed, device=device
    )

    # (2) Fisher on uncertain subset (band by predicted p)
    p_lo_use, p_hi_use = (p_lo, p_hi) if p_lo < p_hi else (p_hi, p_lo)
    p_tr = bb_predict_proba(bb, X_tr, device=device)
    keep = np.where((p_tr >= p_lo_use) & (p_tr <= p_hi_use))[0]
    if keep.size < min_keep:
        order = np.argsort(np.abs(p_tr - 0.5))
        keep = order[:min_keep]
    F = compute_fisher_on_input_x(bb, X_tr[keep], device=device, ridge=1e-6)

    # (3) CBM
    cbm, cbm_acc, cbm_mse = train_cbm_linear_minibatch(
        X_tr, Ck_tr, y_tr,
        X_te, Ck_te, y_te,
        epochs=30, batch_size=512, lr=1e-3,
        lambda_c=1.0, seed=seed, device=device
    )
    Wc = cbm.concept.weight.detach().cpu().numpy()

    with torch.no_grad():
        Xv = torch.tensor(X_te, dtype=torch.float32, device=device)
        c_hat_te, _ = cbm(Xv)
        C_hat_te = c_hat_te.detach().cpu().numpy()

    # (4) SAE
    sae = train_sae_minibatch(
        X_tr,
        K=K_sae, epochs=60, batch_size=512, lr=2e-3, l1=1e-3,
        seed=seed, device=device
    )
    D = sae.D.detach().cpu().numpy()  # (K, r)

    # (5) S matrices
    S_f = fisher_cosine_matrix(Wc, D, F)
    S_e = euclid_cosine_matrix(Wc, D)

    # (6) Z matrices
    Z_f = fisher_cosine_self(Wc, F)
    Z_e = euclid_cosine_self(Wc)

    # ============================================================
    # NEW: C3 prediction from frustrated atoms vs matched non-frustrated atoms
    # ============================================================

    # -------------------------------
    # PATCH: matched-min sampling
    # -------------------------------
    # OLD behavior: if n_frust > n_nonfrust, non_frust_idx becomes empty,
    # leading to constant-mean baseline (R^2 ~ 0) for "non-frust".
    #
    # NEW behavior: use m = min(n_frust, n_nonfrust) and sample m from each side,
    # so both regressions have the same number of features.
    frust_idx_full = _select_frustrated_atoms_pair12(S_f, Z_f, tiny=1e-12)
    n_frust_full = int(frust_idx_full.size)

    all_idx = np.arange(D.shape[0], dtype=np.int64)
    non_frust_pool = np.setdiff1d(all_idx, frust_idx_full, assume_unique=False)
    n_nonfrust_pool = int(non_frust_pool.size)

    rng_atoms = np.random.default_rng(seed + 99991)
    m_match = int(min(n_frust_full, n_nonfrust_pool))

    if m_match > 0:
        frust_idx = rng_atoms.choice(frust_idx_full, size=m_match, replace=False)
        non_frust_idx = rng_atoms.choice(non_frust_pool, size=m_match, replace=False)
    else:
        frust_idx = np.array([], dtype=np.int64)
        non_frust_idx = np.array([], dtype=np.int64)

    PhiF_tr = _project_X_onto_atoms(X_tr, D, frust_idx)
    PhiF_te = _project_X_onto_atoms(X_te, D, frust_idx)
    PhiN_tr = _project_X_onto_atoms(X_tr, D, non_frust_idx)
    PhiN_te = _project_X_onto_atoms(X_te, D, non_frust_idx)

    # Ridge predictions
    yhatF_te, _ = _ridge_predict(PhiF_tr, C3_tr, PhiF_te, lam=ridge_lam_c3)
    yhatN_te, _ = _ridge_predict(PhiN_tr, C3_tr, PhiN_te, lam=ridge_lam_c3)

    C3_mse_frust, C3_r2_frust = _mse_r2(C3_te, yhatF_te)
    C3_mse_nonfrust, C3_r2_nonfrust = _mse_r2(C3_te, yhatN_te)

    # (7) pair frustration metrics
    mf = metrics_from_S_trimmed(S_f)
    me = metrics_from_S_trimmed(S_e)
    pairF = pair_soft_frustration_metric(S_f, Z_f, eps=pair_soft_eps)
    pairE = pair_soft_frustration_metric(S_e, Z_e, eps=pair_soft_eps)

    # (8) geometry difference
    Sd = frob_abs_rel(S_f, S_e)

    # (9) faithfulness vs sample B_known (2x2 vs 2x2)
    Cov_hat = cov_matrix(C_hat_te)
    Corr_hat = corr_from_cov(Cov_hat)
    Corr_B = corr_from_cov(B_known)
    cov_diff = frob_abs_rel(Cov_hat, B_known)
    corr_diff = frob_abs_rel(Corr_hat, Corr_B)

    return {
        "scenario": int(nu),
        "seed": int(seed),
        "k_known": int(k_known),

        "n": int(n),
        "r": int(r),
        "sigma_x": float(sigma_x),
        "d_max": float(d_max),
        "R_task": float(R_task),

        "p_lo": float(p_lo_use),
        "p_hi": float(p_hi_use),
        "F_keep_n": int(len(keep)),

        "bb_acc": float(bb_acc),
        "cbm_acc": float(cbm_acc),
        "cbm_mse": float(cbm_mse),

        "F_frac_best_abs_negative": float(mf["frac_best_abs_negative"]),
        "F_mean_best_abs_signed": float(mf["mean_best_abs_signed"]),
        "F_frac_margin_negative": float(mf["frac_margin_negative"]),

        "E_frac_best_abs_negative": float(me["frac_best_abs_negative"]),
        "E_mean_best_abs_signed": float(me["mean_best_abs_signed"]),
        "E_frac_margin_negative": float(me["frac_margin_negative"]),

        "F_pair_raw_mean": float(pairF["pair_raw_mean"]),
        "F_pair_raw_max": float(pairF["pair_raw_max"]),
        "E_pair_raw_mean": float(pairE["pair_raw_mean"]),
        "E_pair_raw_max": float(pairE["pair_raw_max"]),

        "F_pair_soft_mean": float(pairF["pair_soft_mean"]),
        "F_pair_soft_max": float(pairF["pair_soft_max"]),
        "F_mean_abs_Z": float(pairF["mean_abs_Z"]),
        "E_pair_soft_mean": float(pairE["pair_soft_mean"]),
        "E_pair_soft_max": float(pairE["pair_soft_max"]),
        "E_mean_abs_Z": float(pairE["mean_abs_Z"]),
        "pair_soft_eps": float(pairF["pair_soft_eps"]),

        "S_frob_abs": float(Sd["frob_abs"]),
        "S_frob_rel": float(Sd["frob_rel"]),

        "Cov_frob_abs": float(cov_diff["frob_abs"]),
        "Cov_frob_rel": float(cov_diff["frob_rel"]),
        "Corr_frob_abs": float(corr_diff["frob_abs"]),
        "Corr_frob_rel": float(corr_diff["frob_rel"]),

        # NEW: C3 prediction from SAE projections (with matched-min sampling)
        "n_frust_atoms_full": int(n_frust_full),      # total frust atoms found
        "n_nonfrust_pool": int(n_nonfrust_pool),      # total non-frust available
        "n_atoms_matched": int(m_match),              # used on EACH side

        # Backwards-compatible key (now means "USED frust atoms", not "ALL frust atoms")
        "n_frust_atoms": int(m_match),

        "C3_ridge_lam": float(ridge_lam_c3),
        "C3_mse_frust_atoms": float(C3_mse_frust),
        "C3_r2_frust_atoms": float(C3_r2_frust),
        "C3_mse_nonfrust_atoms": float(C3_mse_nonfrust),
        "C3_r2_nonfrust_atoms": float(C3_r2_nonfrust),
    }

def run_sweep(
    *,
    scenarios,
    seeds,
    device="cpu",
    sigma_x=0.3,
    p_lo=0.2,
    p_hi=0.8,
    min_keep=200,
    r=64,
    n=8000,
    K_sae=60,
    pair_soft_eps=1e-6,
    d_max=1.0,        # ### CHANGED ### default now 1.0
    R_task=0.75,
    ridge_lam_c3=1e-3,
):
    rows = []
    total = len(scenarios) * len(seeds)
    t = 0

    for sc in scenarios:
        for seed in seeds:
            t += 1
            out = run_one(
                nu=int(sc),
                alpha=+1.0,        # unused
                omega=0.0,         # unused
                seed=int(seed),
                k_known=2,
                device=device,
                sigma_x=sigma_x,
                p_lo=p_lo,
                p_hi=p_hi,
                min_keep=min_keep,
                r=r,
                n=n,
                K_sae=K_sae,
                pair_soft_eps=pair_soft_eps,
                d_max=d_max,
                R_task=R_task,
                ridge_lam_c3=ridge_lam_c3,
            )
            rows.append(out)

            sc_name = "cyl" if out["scenario"] == 0 else "sph"
            print(
                f"[{t:03d}/{total:03d}] {sc_name:3s} seed={out['seed']:2d} keep={out['F_keep_n']:4d} | "
                f"bb={out['bb_acc']:.3f} cbm={out['cbm_acc']:.3f} mse={out['cbm_mse']:.3f} | "
                f"Gamma_F(raw_mean)={out['F_pair_raw_mean']:.4f} | "
                f"nF(full/match)={out['n_frust_atoms_full']:2d}/{out['n_atoms_matched']:2d} "
                f"C3R2(F/N)={out['C3_r2_frust_atoms']:.3f}/{out['C3_r2_nonfrust_atoms']:.3f}"
            )

    return rows

def summarize(rows):
    rows = list(rows)
    if not rows:
        print("No rows to summarise.")
        return

    def _stats(vals):
        vals = np.asarray(vals, dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1) if np.sum(np.isfinite(vals)) > 1 else 0.0)

    for sc in [0, 1]:
        subset = [r for r in rows if int(r["scenario"]) == sc]
        name = "cylinder dig" if sc == 0 else "sphere dig"
        if not subset:
            print(f"{name}: no runs")
            continue

        gf_mean, gf_sd = _stats([r["F_pair_raw_mean"] for r in subset])
        bb_mean, bb_sd = _stats([r["bb_acc"] for r in subset])
        cbm_mean, cbm_sd = _stats([r["cbm_acc"] for r in subset])

        # "n_frust_atoms" now means matched/used atoms (see patch).
        nF_mean, nF_sd = _stats([r["n_frust_atoms"] for r in subset])
        c3F_mean, c3F_sd = _stats([r["C3_r2_frust_atoms"] for r in subset])
        c3N_mean, c3N_sd = _stats([r["C3_r2_nonfrust_atoms"] for r in subset])

        # extra bookkeeping summaries (optional)
        nFfull_mean, nFfull_sd = _stats([r.get("n_frust_atoms_full", np.nan) for r in subset])
        nMatch_mean, nMatch_sd = _stats([r.get("n_atoms_matched", np.nan) for r in subset])

        print(f"\n=== {name} (n_runs={len(subset)}) ===")
        print(f"BB acc                  : {bb_mean:.3f} ± {bb_sd:.3f}")
        print(f"CBM acc                 : {cbm_mean:.3f} ± {cbm_sd:.3f}")
        print(f"Gamma_F (raw mean)      : {gf_mean:.4f} ± {gf_sd:.4f}")
        print(f"# frust atoms (FULL)     : {nFfull_mean:.2f} ± {nFfull_sd:.2f}")
        print(f"# atoms matched (USED)   : {nMatch_mean:.2f} ± {nMatch_sd:.2f}")
        print(f"C3 R^2 (frust atoms)     : {c3F_mean:.3f} ± {c3F_sd:.3f}")
        print(f"C3 R^2 (non-frust match) : {c3N_mean:.3f} ± {c3N_sd:.3f}")