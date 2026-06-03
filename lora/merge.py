"""Post-training LoRA merge + LoRA-only state_dict save/load.

Merge: W0 <- W0 + (alpha/r)*B@A -> afterwards inference runs like a plain model
(no extra latency). Save: only lora_A/lora_B parameters (a few MB) -> checkpoints/.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .layer import LoRALinear


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Absorb every LoRALinear's dW into base.weight and zero out the LoRA path.

    After merging, lora_B is set to 0 so the forward output stays identical
    (prevents double-merging).
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = module.delta_weight().to(module.base.weight.dtype)
            module.base.weight.add_(delta)
            module.lora_B.zero_()
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the LoRA parameters (lora_A/lora_B) as CPU tensors."""
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def save_lora(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), path)


def load_lora(model: nn.Module, path: str | Path) -> nn.Module:
    """Load saved LoRA parameters into an (already injected) model."""
    saved = torch.load(path, map_location="cpu")
    own = dict(model.named_parameters())
    missing = set(saved) - set(own)
    if missing:
        raise KeyError(f"LoRA keys not in model: {sorted(missing)} -- different inject config?")
    for name, tensor in saved.items():
        target = own[name]
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
    return model
