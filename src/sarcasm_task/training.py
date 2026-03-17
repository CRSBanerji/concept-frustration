
"""Training and inference utilities for the sarcasm task."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import BlackBoxMLP, LinearCBM, SparseAE
from .metrics import accuracy_from_logits


def bb_predict_proba(bb: BlackBoxMLP, X: np.ndarray, *, device="cpu") -> np.ndarray:
    bb.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = bb(Xt)
    return torch.sigmoid(logits).detach().cpu().numpy()


def compute_fisher_on_input_x(bb: BlackBoxMLP, X: np.ndarray, *, device="cpu", ridge=1e-6):
    bb.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)

    logits, z, _ = bb(Xt, return_hidden=True)
    p = torch.sigmoid(logits)
    s = p * (1 - p)
    m = (z > 0).float()

    W_H = bb.fc1.weight
    w_l = bb.fc2.weight.squeeze(0)

    U = m * w_l.unsqueeze(0)
    G = U @ W_H

    Fm = (G.T @ (G * s.unsqueeze(1))) / float(X.shape[0])
    Fm = Fm + ridge * torch.eye(Fm.shape[0], device=device)
    return Fm.detach().cpu().numpy()


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
        for s0 in range(0, n, batch_size):
            b = idx[s0:s0+batch_size]
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
        for s0 in range(0, n, batch_size):
            b = idx[s0:s0+batch_size]
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
        for s0 in range(0, n, batch_size):
            b = idx[s0:s0+batch_size]
            xb = Xt[b]
            opt.zero_grad()
            s_code, xhat = model(xb)
            loss = ((xhat - xb)**2).mean() + l1 * s_code.abs().mean()
            loss.backward()
            opt.step()

    return model
