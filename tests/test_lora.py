"""LoRA core unit tests — instant verification without downloading any model (pytest).

Verifies the correctness of the LoRA formula, injection, and merge using pure
torch without external libraries (formula match, B=0 initial invariance, output
invariance after merge).

    pytest tests/test_lora.py -q
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from lora import (
    count_parameters,
    inject_lora,
    mark_only_lora_as_trainable,
    merge_lora,
)
from lora.layer import LoRALinear


class TinyAttn(nn.Module):
    """Minimal module with q_proj/k_proj/v_proj/o_proj (mimics Qwen module names)."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.v_proj(x))


def test_init_delta_is_zero():
    """B=0 init → ΔW=0 at start, forward identical to base."""
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(2, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_forward_matches_formula():
    """forward matches the formula W0·x + (alpha/r)·B·(A·x)."""
    torch.manual_seed(1)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, r=4, alpha=8.0)
    nn.init.normal_(lora.lora_B, std=0.1)  # make B non-trivial
    x = torch.randn(3, 16)
    expected = base(x) + lora.scaling * (x @ lora.lora_A.t() @ lora.lora_B.t())
    assert torch.allclose(lora(x), expected, atol=1e-5)


def test_base_frozen_lora_trainable():
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, r=4, alpha=8.0)
    assert not lora.base.weight.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_inject_and_count():
    model = TinyAttn(16)
    _, replaced = inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=8.0)
    assert set(replaced) == {"q_proj", "v_proj"}
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.k_proj, nn.Linear) and not isinstance(model.k_proj, LoRALinear)

    mark_only_lora_as_trainable(model)
    trainable, total = count_parameters(model)
    # two modules × (A:r×16 + B:16×r) = 2×(64+64)=256
    assert trainable == 256
    assert trainable < total


def test_inject_no_match_raises():
    model = TinyAttn(16)
    try:
        inject_lora(model, ["nonexistent"], r=4, alpha=8.0)
    except ValueError:
        return
    raise AssertionError("ValueError should be raised when there is no match")


def test_merge_preserves_output():
    """forward output after merge is identical to before merge (zero inference latency)."""
    torch.manual_seed(2)
    model = TinyAttn(16)
    inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=8.0)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.1)

    x = torch.randn(5, 16)
    before = model(x).clone()
    merge_lora(model)
    after = model(x)
    assert torch.allclose(before, after, atol=1e-5)
