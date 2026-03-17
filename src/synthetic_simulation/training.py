import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import BlackBoxMLP, LinearCBM, SparseAE

def accuracy_from_logits(logits: torch.Tensor, y_true_np: np.ndarray) -> float:
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)
    return float((preds == y_true_np).mean())

def train_test_split_indices(n: int, train_frac: float, rng: np.random.Generator):
    perm = rng.permutation(n)
    split = int(train_frac * n)
    return perm[:split], perm[split:]

def bb_predict_proba(bb: BlackBoxMLP, X: np.ndarray, *, device="cpu") -> np.ndarray:
    """Return p(y=1|x) for each row of X."""
    bb.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = bb(Xt)
    return torch.sigmoid(logits).detach().cpu().numpy()

def train_bb_minibatch(X_tr, y_tr, X_te, y_te, *, hidden=128, epochs=30, batch_size=512, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    r = X_tr.shape[1]
    model = BlackBoxMLP(r, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_te, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    idx = np.arange(n)

    for _ in range(epochs):
        np.random.shuffle(idx)
        model.train()
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            xb, yb = Xt[b], yt[b]
            opt.zero_grad()
            loss = bce(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        acc_te = accuracy_from_logits(model(Xv), y_te)
    return model, acc_te

def train_cbm_linear_minibatch(X_tr, Ck_tr, y_tr, X_te, Ck_te, y_te, *,
                              epochs=30, batch_size=512, lr=1e-3, lambda_c=1.0, seed=0, device="cpu"):
    """
    BCE(task) + lambda_c * MSE(concepts)
    """
    torch.manual_seed(seed); np.random.seed(seed)
    xdim = X_tr.shape[1]
    k = Ck_tr.shape[1]
    model = LinearCBM(xdim, k).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Ct = torch.tensor(Ck_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)

    Xv = torch.tensor(X_te, dtype=torch.float32, device=device)
    Cv = torch.tensor(Ck_te, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    idx = np.arange(n)

    for _ in range(epochs):
        np.random.shuffle(idx)
        model.train()
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            xb, cb, yb = Xt[b], Ct[b], yt[b]
            opt.zero_grad()
            c_hat, logit = model(xb)
            loss = bce(logit, yb) + lambda_c * mse(c_hat, cb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        c_hat_te, logit_te = model(Xv)
        acc_te = accuracy_from_logits(logit_te, y_te)
        mse_te = float(((c_hat_te - Cv)**2).mean().cpu().item())

    return model, acc_te, mse_te

def train_cbm_hard_two_stage_predicted(
    X_tr, Ck_tr, y_tr,
    X_te, Ck_te, y_te,
    *,
    concept_epochs=30,
    task_epochs=30,
    batch_size=512,
    lr_concept=1e-3,
    lr_task=1e-3,
    seed=0,
    device="cpu",
):
    """
    HARD CBM:
      Stage A: train concept head only (MSE on concepts).
      Stage B: freeze concept head; train task head only (BCE on y) using predicted concepts.

    Returns: model, acc_test, concept_mse_test
    """
    torch.manual_seed(seed); np.random.seed(seed)

    xdim = X_tr.shape[1]
    k = Ck_tr.shape[1]
    model = LinearCBM(xdim, k).to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Ct = torch.tensor(Ck_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)

    Xv = torch.tensor(X_te, dtype=torch.float32, device=device)
    Cv = torch.tensor(Ck_te, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    idx = np.arange(n)

    # ---- Stage A: concept head only ----
    for p in model.task.parameters():
        p.requires_grad_(False)
    for p in model.concept.parameters():
        p.requires_grad_(True)

    opt_c = optim.Adam(model.concept.parameters(), lr=lr_concept)

    for _ in range(concept_epochs):
        np.random.shuffle(idx)
        model.train()
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            xb, cb = Xt[b], Ct[b]
            opt_c.zero_grad()
            c_hat = model.concept(xb)
            loss = mse(c_hat, cb)
            loss.backward()
            opt_c.step()

    # ---- Stage B: task head only (freeze concept head) ----
    for p in model.concept.parameters():
        p.requires_grad_(False)
    for p in model.task.parameters():
        p.requires_grad_(True)

    opt_t = optim.Adam(model.task.parameters(), lr=lr_task)

    for _ in range(task_epochs):
        np.random.shuffle(idx)
        model.train()
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            xb, yb = Xt[b], yt[b]
            opt_t.zero_grad()
            with torch.no_grad():
                c_hat = model.concept(xb)  
            logit = model.task(c_hat).squeeze(-1)
            loss = bce(logit, yb)
            loss.backward()
            opt_t.step()

    # ---- Eval ----
    model.eval()
    with torch.no_grad():
        c_hat_te = model.concept(Xv)
        logit_te = model.task(c_hat_te).squeeze(-1)
        acc_te = accuracy_from_logits(logit_te, y_te)
        mse_te = float(((c_hat_te - Cv)**2).mean().cpu().item())

    return model, acc_te, mse_te

def train_cbm_hard_two_stage_ground_truth(
    X_tr, Ck_tr, y_tr,
    X_te, Ck_te, y_te,
    *,
    concept_epochs=30,
    task_epochs=30,
    batch_size=512,
    lr_concept=1e-3,
    lr_task=1e-3,
    seed=0,
    device="cpu",
):
    """
    HARD CBM (2-stage, sequential / split training):

      Stage A (concept learning):
        - Train concept head ONLY: X -> C_known
        - Loss: MSE(c_hat, c_known)

      Stage B (task learning, concept head frozen):
        - Freeze concept head
        - Train task head ONLY: C_known -> y   **USING GROUND-TRUTH CONCEPTS**
        - Loss: BCE(y_logit, y)
    Returns: model, acc_test, concept_mse_test
    """
    torch.manual_seed(seed); np.random.seed(seed)

    xdim = X_tr.shape[1]
    k = Ck_tr.shape[1]
    model = LinearCBM(xdim, k).to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Ct = torch.tensor(Ck_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)

    Xv = torch.tensor(X_te, dtype=torch.float32, device=device)
    Cv = torch.tensor(Ck_te, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    idx = np.arange(n)

    # ---- Stage A: concept head only ----
    for p in model.task.parameters():
        p.requires_grad_(False)
    for p in model.concept.parameters():
        p.requires_grad_(True)

    opt_c = optim.Adam(model.concept.parameters(), lr=lr_concept)

    for _ in range(concept_epochs):
        np.random.shuffle(idx)
        model.train()
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            xb, cb = Xt[b], Ct[b]
            opt_c.zero_grad()
            c_hat = model.concept(xb)
            loss = mse(c_hat, cb)
            loss.backward()
            opt_c.step()

    # ---- Stage B: task head only (freeze concept head) ----
    for p in model.concept.parameters():
        p.requires_grad_(False)
    for p in model.task.parameters():
        p.requires_grad_(True)

    opt_t = optim.Adam(model.task.parameters(), lr=lr_task)

    for _ in range(task_epochs):
        np.random.shuffle(idx)
        model.train()
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]

            cb, yb = Ct[b], yt[b]

            opt_t.zero_grad()
            logit = model.task(cb).squeeze(-1)
            loss = bce(logit, yb)
            loss.backward()
            opt_t.step()

    model.eval()
    with torch.no_grad():
        c_hat_te = model.concept(Xv)
        logit_te = model.task(c_hat_te).squeeze(-1)
        acc_te = accuracy_from_logits(logit_te, y_te)
        mse_te = float(((c_hat_te - Cv)**2).mean().cpu().item())

    return model, acc_te, mse_te

def train_sae_minibatch(X_tr, *, K=60, epochs=60, batch_size=512, lr=2e-3, l1=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    xdim = X_tr.shape[1]
    model = SparseAE(xdim, K).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    idx = np.arange(n)

    for _ in range(epochs):
        np.random.shuffle(idx)
        model.train()
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            xb = Xt[b]
            opt.zero_grad()
            s_code, xhat = model(xb)
            loss = ((xhat - xb)**2).mean() + l1 * s_code.abs().mean()
            loss.backward()
            opt.step()

    return model
