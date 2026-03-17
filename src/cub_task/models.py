import torch
import torch.nn as nn

class BlackBoxMLP(nn.Module):
    def __init__(self, r: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(r, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x, return_hidden: bool = False):
        z = self.fc1(x)
        h = self.act(z)
        logit = self.fc2(h).squeeze(-1)
        if return_hidden:
            return logit, z, h
        return logit

class LinearCBM(nn.Module):
    def __init__(self, xdim: int, k_known: int):
        super().__init__()
        self.concept = nn.Linear(xdim, k_known)
        self.task = nn.Linear(k_known, 1)

    def forward(self, x):
        c = self.concept(x)
        y = self.task(c).squeeze(-1)
        return c, y

class SparseAE(nn.Module):
    def __init__(self, xdim: int, K: int):
        super().__init__()
        self.W = nn.Linear(xdim, K, bias=True)
        self.D = nn.Parameter(torch.randn(K, xdim) * 0.02)

    def forward(self, x):
        s = torch.relu(self.W(x))
        x_hat = s @ self.D
        return s, x_hat
