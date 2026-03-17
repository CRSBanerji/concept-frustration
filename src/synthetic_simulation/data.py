import numpy as np

def make_random_spd(k: int, rng: np.random.Generator, jitter: float = 1e-6) -> np.ndarray:
    A = rng.normal(size=(k, k))
    B = A @ A.T + jitter * np.eye(k)
    return (B + B.T) / 2.0

def is_spd(B: np.ndarray, tol: float = 1e-10) -> bool:
    eigs = np.linalg.eigvalsh((B + B.T) / 2.0)
    return float(np.min(eigs)) > tol

def _ensure_spd(B: np.ndarray, *, tol: float = 1e-10, jitter0: float = 1e-8) -> np.ndarray:
    """Symmetrize and add increasing diagonal jitter until SPD."""
    B = (B + B.T) / 2.0
    jitter = float(jitter0)
    while not is_spd(B, tol=tol):
        B = B + jitter * np.eye(B.shape[0])
        jitter *= 10.0
    return (B + B.T) / 2.0

def sample_pair_assignment(
    B_known: np.ndarray,
    k_unknown: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Seed-fixed assignment u -> (i(u), j(u), b_{i(u)j(u)}).

    Returns:
      I_idx : (k_unknown,) int
      J_idx : (k_unknown,) int
      b_ij  : (k_unknown,) float  where b_ij[u] = B_known[I_idx[u], J_idx[u]]

    """
    B_known = (B_known + B_known.T) / 2.0
    k_known = B_known.shape[0]
    assert k_unknown > 0

    edges = []
    for i in range(k_known):
        for j in range(i + 1, k_known):
            b = float(B_known[i, j])
            # if exactly zero (measure zero with random SPD), skip it
            if b != 0.0:
                edges.append((i, j, b))

    if len(edges) < k_unknown:
        raise ValueError(
            f"Not enough nonzero known-pairs to assign: need k_unknown={k_unknown}, "
            f"but only {len(edges)} nonzero edges in B_known."
        )

    rng.shuffle(edges)
    chosen = edges[:k_unknown]

    I_idx = np.array([e[0] for e in chosen], dtype=np.int64)
    J_idx = np.array([e[1] for e in chosen], dtype=np.int64)
    b_ij  = np.array([e[2] for e in chosen], dtype=np.float64)

    return I_idx, J_idx, b_ij

def build_M_alpha_from_assignment(
    *,
    I_idx: np.ndarray,
    J_idx: np.ndarray,
    b_ij: np.ndarray,
    k_known: int,
    k_unknown: int,
    alpha: float,
    alpha_strength: float = 1.0,
) -> np.ndarray:
    """
    Construct M(alpha)

    - Exactly two nonzeros per column u: rows i(u) and j(u)
    - α sign selects regime:
        α > 0 : frustrating
        α < 0 : consistent
        α = 0 : independence (M=0)
    - |α| controls interaction strength (overall scale)

    using b = b_{i(u)j(u)}:
      Frustrating (α>0):
        if b>0:  M[i,u]= b|α|,  M[j,u]= -b|α|
        if b<0:  M[i,u]= b|α|,  M[j,u]=  b|α|
      Consistent (α<0):
        if b>0:  M[i,u]= b|α|,  M[j,u]=  b|α|
        if b<0:  M[i,u]= b|α|,  M[j,u]= -b|α|
    """
    a = float(abs(alpha) * abs(alpha_strength))
    if a == 0.0:
        return np.zeros((k_known, k_unknown), dtype=np.float64)

    M = np.zeros((k_known, k_unknown), dtype=np.float64)

    # Vectorised sign checks
    pos = (b_ij > 0.0)
    neg = (b_ij < 0.0)

    # Start by setting M[i,u] = b|α| for all u
    M[I_idx, np.arange(k_unknown)] = b_ij * a

    if alpha > 0.0:
        # Frustrating: j gets (-b|α|) if b>0; j gets (b|α|) if b<0
        # i.e. j = -b for pos, j = +b for neg
        j_vals = np.empty_like(b_ij)
        j_vals[pos] = -b_ij[pos] * a
        j_vals[neg] =  b_ij[neg] * a
        M[J_idx, np.arange(k_unknown)] = j_vals
    else:
        # Consistent: j gets (+b|α|) if b>0; j gets (-b|α|) if b<0
        # i.e. j = +b for pos, j = -b for neg
        j_vals = np.empty_like(b_ij)
        j_vals[pos] =  b_ij[pos] * a
        j_vals[neg] = -b_ij[neg] * a
        M[J_idx, np.arange(k_unknown)] = j_vals

    return M

def build_B_alpha_from_components(
    B_known: np.ndarray,
    B_temp: np.ndarray,
    assignment: tuple[np.ndarray, np.ndarray, np.ndarray],  # (I_idx, J_idx, b_ij)
    alpha: float,
    *,
    alpha_strength: float = 1.0,
    ensure_spd: bool = True,
) -> np.ndarray:
    """
    Given seed-fixed (B_known, B_temp, assignment), build B(alpha) via Schur completion:
      B_uu(alpha) = B_temp + M(alpha)^T B_known^{-1} M(alpha)
      B(alpha)    = [[B_known, M(alpha)],
                     [M(alpha)^T, B_uu(alpha)]]

    This preserves:
      Cov(C_known) = B_known
      Cov(C_unknown | C_known) = B_temp
    and enforces SPD by construction (and optional numerical jitter).
    """
    alpha = float(np.clip(alpha, -1.0, 1.0))

    B_known = (B_known + B_known.T) / 2.0
    B_temp  = (B_temp  + B_temp.T)  / 2.0

    k_known = B_known.shape[0]
    k_unknown = B_temp.shape[0]
    I_idx, J_idx, b_ij = assignment

    M = build_M_alpha_from_assignment(
        I_idx=I_idx, J_idx=J_idx, b_ij=b_ij,
        k_known=k_known, k_unknown=k_unknown,
        alpha=alpha, alpha_strength=alpha_strength
    )

    Binv = np.linalg.inv(B_known)
    B_uu = B_temp + (M.T @ Binv @ M)
    B_uu = (B_uu + B_uu.T) / 2.0

    B = np.block([
        [B_known, M],
        [M.T,     B_uu],
    ])
    B = (B + B.T) / 2.0

    if ensure_spd:
        B = _ensure_spd(B, tol=1e-10, jitter0=1e-10)

    return B

def sample_B_components_for_seed(
    k: int,
    k_known: int,
    seed: int,
    *,
    jitter0: float = 1e-8,
    alpha_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Sample seed-fixed components once per (seed, k_known):
      - B_known (SPD)
      - B_temp  (SPD)
      - assignment = (I_idx, J_idx, b_ij)  one known-pair per unknown column
      - alpha_strength  (stored for convenience)
    """
    assert 0 < k_known < k
    k_unknown = k - k_known

    rng = np.random.default_rng(seed)

    B_known = _ensure_spd(make_random_spd(k_known, rng), tol=1e-10, jitter0=jitter0)
    B_temp  = _ensure_spd(make_random_spd(k_unknown, rng), tol=1e-10, jitter0=jitter0)

    assignment = sample_pair_assignment(B_known=B_known, k_unknown=k_unknown, rng=rng)

    return B_known, B_temp, assignment, float(alpha_strength)

def generate_toy_dataset_concepts_first(
    n: int,
    r: int,
    k: int,
    k_known: int,
    sigma_x: float,
    sigma_y: float,
    omega: float,
    seed: int = 0,
    alpha: float = 0.0,
    A_scale: str | None = "none",
    *,
    B_components=None,
    alpha_strength: float = 1.0,
):
    
    assert r > 0 and k > 0 and n > 0
    assert 0 < k_known < k

    omega = float(np.clip(omega, 0.0, 1.0))
    alpha = float(np.clip(alpha, -1.0, 1.0))

    if B_components is None:
        B_known, B_temp, assignment, alpha_strength_used = sample_B_components_for_seed(
            k=k, k_known=k_known, seed=seed, alpha_strength=alpha_strength
        )
    else:
        # Expect the new 4-tuple returned by sample_B_components_for_seed
        B_known, B_temp, assignment, alpha_strength_used = B_components

    B = build_B_alpha_from_components(
        B_known=B_known,
        B_temp=B_temp,
        assignment=assignment,
        alpha=alpha,
        alpha_strength=alpha_strength_used,
        ensure_spd=True,
    )

    rng = np.random.default_rng(seed)

    # Sample C ~ N(0,B)
    Lb = np.linalg.cholesky((B + B.T) / 2.0 + 1e-12 * np.eye(k))
    C = rng.normal(size=(n, k)) @ Lb.T

    # Mixing matrix A and X
    A = rng.normal(size=(r, k))

    if A_scale is None:
        pass
    else:
        A_scale_l = str(A_scale).strip().lower()
        if A_scale_l in ("unit_var", "unitvar"):
            for j in range(r):
                aj = A[j]
                v = float(aj @ B @ aj)
                A[j] = aj / np.sqrt(v + 1e-12)
        elif A_scale_l in ("none", "no", "false"):
            pass
        else:
            raise ValueError(f"Unknown A_scale={A_scale!r}. Use 'unit_var' or 'none'/None.")

    X = C @ A.T + rng.normal(size=(n, r), scale=float(sigma_x))

    # Task
    w = rng.normal(size=(k,))
    w_star = w.copy()
    w_star[:k_known] *= (1.0 - omega)
    w_star[k_known:] *= omega

    eta = rng.normal(size=(n,), scale=float(sigma_y))
    score = C @ w_star + eta
    y = (score > 0).astype(np.int64)

    return (
        X.astype(np.float32),
        C.astype(np.float32),
        y,
        B.astype(np.float32),
        A.astype(np.float32),
        w.astype(np.float32),
        w_star.astype(np.float32),
        (B_known.astype(np.float32), B_temp.astype(np.float32), assignment, float(alpha_strength_used)),
    )
