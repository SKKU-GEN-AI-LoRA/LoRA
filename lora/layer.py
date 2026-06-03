"""LoRALinear -- from-scratch implementation of the LoRA low-rank adaptation layer
(Hu et al., 2022).

    W = W0 + dW = W0 + (alpha / r) * B @ A
        W0 in R^{out x in}  (frozen, pretrained weight)
        A  in R^{r x in}    (Gaussian init)
        B  in R^{out x r}   (zero init)
    -> at the start of training dW = B@A = 0, so the pretrained behavior is preserved.

forward adds the base path and the LoRA path:
    out = W0*x + (alpha/r) * B*(A*x)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap an existing ``nn.Linear`` with a frozen weight + a low-rank trainable path."""

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        init: str = "gaussian",
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank r must be positive: {r}")

        # Freeze the pretrained weight (and bias).
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        self.in_features = base.in_features
        self.out_features = base.out_features

        device, dtype = base.weight.device, base.weight.dtype
        # A: (r, in), B: (out, r) -- same device/dtype as base.weight.
        self.lora_A = nn.Parameter(
            torch.empty(r, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, r, device=device, dtype=dtype)
        )
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._init_lora(init)

    def _init_lora(self, init: str) -> None:
        """Initialize A (B is always 0 -> dW=0 at the start)."""
        if init == "gaussian":
            # Hu et al.: A ~ N(0, sigma^2). sigma=1/r keeps it stable regardless of scaling.
            nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / self.r)
        elif init == "kaiming":
            # Alternative init (kaiming uniform).
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            raise ValueError(f"unknown init: {init!r} (gaussian|kaiming)")
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # F.linear(x, A) = x @ A.T -> (.., r); apply B again -> (.., out)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """Return dW = (alpha/r)*B@A for merging/verification (out x in)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )
