"""Experiment configuration loading (YAML -> dataclass). Shared by scripts/ and eval/.

One Config expresses a point in the (rank x target_modules x alpha x method)
variation matrix. One YAML file = one experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # -- Base model --------------------------------------------------
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # fallback: 0.5B-Instruct

    # -- Method: lora | full | zeroshot ------------------------------
    method: str = "lora"

    # -- LoRA hyperparameters ----------------------------------------
    r: int = 8
    alpha: float = 16.0  # scaling = alpha / r
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.0
    lora_init: str = "gaussian"  # gaussian | kaiming

    # -- Data --------------------------------------------------------
    train_dataset: str = "medmcqa"
    train_subset: int | None = 10_000  # None = full (182,822)
    eval_datasets: list[str] = field(
        default_factory=lambda: ["medmcqa", "medqa", "pubmedqa"]
    )
    eval_subset: int | None = None  # None = full eval set
    max_length: int = 512

    # -- Training ----------------------------------------------------
    lr: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 1
    max_steps: int | None = None  # if set, overrides epochs
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    grad_checkpointing: bool = False  # for tight memory
    log_every: int = 10
    eval_every: int | None = None  # step-wise eval (None = only after training)

    # -- Misc --------------------------------------------------------
    seed: int = 42
    output_dir: str = "checkpoints/run"
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path, **overrides: Any) -> Config:
    """Read a YAML file and apply CLI overrides to build a Config."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    known = {f for f in Config.__dataclass_fields__}  # noqa: C416
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**raw)
