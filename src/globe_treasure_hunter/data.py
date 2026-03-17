"""
Flat vs Spherical "Earth" Concept-Frustration Simulation
========================================================

SPHERE (scenario=1):
  u ~ Unif(S^2)
  d ~ Unif[0,d_max)
  p = (1-d) u

CYLINDER (scenario=0):
  (x_u,y_u) ~ Unif(unit disk)
  d ~ Unif[0,d_max)
  p = (x_u, y_u, -d)

Concepts (surface-parallel distances to NP and SP, and depth)
  - We restrict depth to d in [0, 1] via d_max=1.0.
  - Cylinder (disk at z=-d): "parallel-to-surface" distance is straight-line distance IN THE PLANE z=-d:
        C1 = ||(x,y) - (0, +1)||_2
        C2 = ||(x,y) - (0, -1)||_2
  - Sphere (sphere of radius r=1-d): "parallel-to-surface" distance is GEODESIC distance on that sphere:
        r = 1 - d
        North pole at depth d: N_d = (0, +r, 0)
        South pole at depth d: S_d = (0, -r, 0)
        C1 = r * arccos( (p·N_d) / r^2 ) = r * arccos(u_y)
        C2 = r * arccos( (p·S_d) / r^2 ) = r * arccos(-u_y)
  - C3 = d (unknown concept)

Task:
  y = 1{ ||p - E||_2 < R_task }, where E=(1,0,0)
"""
import numpy as np

NORTH   = np.array([0.0,  1.0, 0.0], dtype=np.float32)
SOUTH   = np.array([0.0, -1.0, 0.0], dtype=np.float32)
EQUATOR = np.array([1.0,  0.0, 0.0], dtype=np.float32)

def sample_uniform_disk(n: int, rng: np.random.Generator):
    """Uniform samples over the unit disk via polar coordinates."""
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rr = np.sqrt(rng.uniform(0.0, 1.0, size=n))
    x = (rr * np.cos(theta)).astype(np.float32)
    y = (rr * np.sin(theta)).astype(np.float32)
    return x, y

def sample_uniform_sphere_directions(n: int, rng: np.random.Generator):
    """Uniform direction samples on S^2 via normal->normalize."""
    v = rng.normal(size=(n, 3)).astype(np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

def sample_depth(n: int, rng: np.random.Generator, *, d_max: float = 1.0) -> np.ndarray:
    #d ~ Unif[0, d_max)
    return rng.uniform(0.0, float(d_max), size=n).astype(np.float32)

def generate_sphere_dig_concepts(n: int, seed: int, *, d_max: float = 1.0, R: float = 0.75):
    """
    Sphere setup:
      u ~ Unif(S^2), d ~ Unif[0,d_max), p=(1-d)u
    Concepts:
      Let r = 1 - d (sphere radius at depth d).
      Define "surface-parallel" distances as GEODESIC distances on the radius-r sphere to:
        N_d = (0, +r, 0), S_d = (0, -r, 0).
      Then for p=r*u:
        C1 = r * arccos( (p·N_d)/r^2 ) = r * arccos(u_y)
        C2 = r * arccos( (p·S_d)/r^2 ) = r * arccos(-u_y)
      C3 = d
    Task:
      y = 1{ ||p - EQUATOR||_2 < R }
    """
    rng = np.random.default_rng(seed)
    u = sample_uniform_sphere_directions(n, rng)          
    d = sample_depth(n, rng, d_max=d_max)                 
    r = (1.0 - d).astype(np.float32)                      
    P = r[:, None] * u                                    

    # Since (p·N_d)/r^2 = u_y and (p·S_d)/r^2 = -u_y
    uy = u[:, 1].astype(np.float32)
    uy = np.clip(uy, -1.0, 1.0)

    c1 = (r * np.arccos(uy)).astype(np.float32)
    c2 = (r * np.arccos(-uy)).astype(np.float32)
    c3 = d.astype(np.float32)

    dist_to_e = np.linalg.norm(P - EQUATOR[None, :], axis=1).astype(np.float32)
    y_bin = (dist_to_e < float(R)).astype(np.int64)

    C = np.stack([c1, c2, c3], axis=1).astype(np.float32)
    return C, y_bin

def generate_cylinder_dig_concepts(n: int, seed: int, *, d_max: float = 1.0, R: float = 0.75):
    """
    Cylinder setup:
      (x_u,y_u) ~ Unif(unit disk), d ~ Unif[0,d_max), p=(x_u,y_u,-d)

    Concepts:
      Define "surface-parallel" distances on the disk plane z=-d to points:
        N_d = (0, +1, -d), S_d = (0, -1, -d).
      The intrinsic (geodesic) distance on the plane equals the straight-line distance in (x,y):
        C1 = sqrt( (x-0)^2 + (y-1)^2 )
        C2 = sqrt( (x-0)^2 + (y+1)^2 )
      C3 = d

    Task:
      y = 1{ ||p - EQUATOR||_2 < R }
    """
    rng = np.random.default_rng(seed)
    x, y = sample_uniform_disk(n, rng)
    d = sample_depth(n, rng, d_max=d_max)
    z = (-d).astype(np.float32)

    P = np.stack([x, y, z], axis=1).astype(np.float32)

    # Planar distances on disk at fixed depth (z cancels):
    c1 = np.sqrt((x * x) + (y - 1.0) * (y - 1.0)).astype(np.float32)
    c2 = np.sqrt((x * x) + (y + 1.0) * (y + 1.0)).astype(np.float32)
    c3 = d.astype(np.float32)

    dist_to_e = np.linalg.norm(P - EQUATOR[None, :], axis=1).astype(np.float32)
    y_bin = (dist_to_e < float(R)).astype(np.int64)

    C = np.stack([c1, c2, c3], axis=1).astype(np.float32)
    return C, y_bin

def concepts_to_signal(C: np.ndarray, *, r: int = 64, sigma_x: float = 0.3, seed: int = 0):
    """
    X = C @ A^T + eps
    Treat X as the model input / activation proxy (a).
    """
    rng = np.random.default_rng(seed)
    n, k = C.shape
    A = rng.normal(size=(r, k)).astype(np.float32)
    eps = rng.normal(scale=sigma_x, size=(n, r)).astype(np.float32)
    X = (C @ A.T) + eps
    return X.astype(np.float32), A

def generate_geo_dataset(*, scenario: int, n: int, r: int, sigma_x: float, seed: int, d_max: float, R_task: float):
    """
    Return:
      X: (n,r) signal / activation proxy
      C: (n,3) concepts
      y: (n,) labels
      B: (3,3) sample covariance of C
    """
    if int(scenario) == 0:
        C, y = generate_cylinder_dig_concepts(n=n, seed=seed, d_max=d_max, R=R_task)
    elif int(scenario) == 1:
        C, y = generate_sphere_dig_concepts(n=n, seed=seed, d_max=d_max, R=R_task)
    else:
        raise ValueError("scenario must be 0 (cylinder) or 1 (sphere)")

    X, _A = concepts_to_signal(C, r=r, sigma_x=sigma_x, seed=seed + 12345)
    B = np.cov(C, rowvar=False, bias=False).astype(np.float32)
    return X, C, y, B

def train_test_split_indices(n: int, train_frac: float, rng: np.random.Generator):
    perm = rng.permutation(n)
    split = int(train_frac * n)
    return perm[:split], perm[split:]
