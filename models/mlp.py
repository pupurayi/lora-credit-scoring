"""
Three-layer feedforward network for tabular credit scoring, plus a LoRA
adapter wrapper.

Architecture:
    Input -> Linear(d_in, 512) -> BN -> ReLU -> Dropout
          -> Linear(512, 256)   -> BN -> ReLU -> Dropout
          -> Linear(256, 128)   -> BN -> ReLU -> Dropout
          -> Linear(128, 1)

LoRA adapters can be injected into the two largest weight matrices
(d_in -> 512 and 512 -> 256) via apply_lora().
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    d_in: int
    hidden: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.2
    out_dim: int = 1


class TabularMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        prev = cfg.d_in
        # We keep references to the Linear layers we may want to LoRA-wrap.
        self.linears: list[nn.Linear] = []
        for h in cfg.hidden:
            lin = nn.Linear(prev, h)
            self.linears.append(lin)
            layers += [lin, nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(cfg.dropout)]
            prev = h
        self.head = nn.Linear(prev, cfg.out_dim)
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        logit = self.head(h)
        return logit  # raw logit; BCEWithLogitsLoss handles sigmoid

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))


class LoRALinear(nn.Module):
    """
    Wraps an nn.Linear with a low-rank residual update.

        y = W0 x + b  +  (alpha / r) * (B A x)
            ^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^
            frozen         trainable, A ~ N(0, sigma), B initialised to 0

    Parameters of the inner Linear (W0, b) are frozen; only A, B are trainable.
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.linear = linear
        # Freeze the wrapped layer's weights and bias
        for p in self.linear.parameters():
            p.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # A: r x in_features, B: out_features x r
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        # B stays zero so the LoRA delta starts at zero - the model behaves
        # identically to the pre-trained network at step 0.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)                 # (batch, out)
        delta = (x @ self.A.T) @ self.B.T     # (batch, out)
        return base + self.scaling * delta

    def trainable_parameters(self) -> int:
        return self.A.numel() + self.B.numel()


def apply_lora(model: TabularMLP, r: int = 8, alpha: int = 16,
               which: tuple[int, ...] = (0, 1)) -> int:
    """
    Replace the two largest Linear modules with LoRA-wrapped versions.
    `which` indexes into model.linears (default: first two hidden layers,
    which are the largest by parameter count for our architecture).
    Returns total trainable parameter count (LoRA + biases).
    """
    # Freeze ALL base parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Replace selected Linear modules with LoRALinear in the Sequential
    # We need to find the Linear objects inside model.body and swap them.
    swap_targets = {id(model.linears[i]): i for i in which}
    new_modules = []
    for m in model.body:
        if isinstance(m, nn.Linear) and id(m) in swap_targets:
            new_modules.append(LoRALinear(m, r=r, alpha=alpha))
        else:
            new_modules.append(m)
    model.body = nn.Sequential(*new_modules)

    # Re-enable head training too (small layer, common LoRA practice)
    for p in model.head.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
