"""Module-traversal injection logic that replaces selected modules with LoRALinear.

Qwen2.5 module names:
    attention : q_proj, k_proj, v_proj, o_proj
    MLP       : gate_proj, up_proj, down_proj
target_modules is a subset of these leaf names.
"""

from __future__ import annotations

import torch.nn as nn

from .layer import LoRALinear


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    r: int,
    alpha: float,
    dropout: float = 0.0,
    init: str = "gaussian",
) -> tuple[nn.Module, list[str]]:
    """Replace every ``nn.Linear`` matching ``target_modules`` with a LoRALinear.

    Returns:
        (model, list of replaced module paths)
    """
    targets = set(target_modules)
    replaced: list[str] = []

    # Walk named_modules() and inspect each parent's direct children.
    # We mutate parent attributes via setattr, so we access by named_children.
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout, init=init)
                setattr(parent, child_name, lora)
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                replaced.append(full)

    if not replaced:
        raise ValueError(
            f"no nn.Linear matched target_modules={target_modules}. "
            "Check the model module names."
        )
    return model, replaced


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Keep only LoRA parameters (lora_A/lora_B) trainable; freeze the rest."""
    for name, param in model.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts. Efficiency metric."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
