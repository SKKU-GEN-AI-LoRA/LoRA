"""LoRA from scratch -- low-rank adaptation implemented without any library (no peft).

Implemented from scratch:
  - LoRALinear     : wraps nn.Linear with a frozen W0 + a low-rank path
  - inject_lora    : traverse the model and replace selected modules with LoRALinear
  - merge_lora     : absorb BA into W0 after training (zero inference latency)
  - save/load_lora : save only the LoRA parameters (a few MB)
"""

from .inject import (
    count_parameters,
    inject_lora,
    mark_only_lora_as_trainable,
)
from .layer import LoRALinear
from .merge import load_lora, lora_state_dict, merge_lora, save_lora

__all__ = [
    "LoRALinear",
    "inject_lora",
    "mark_only_lora_as_trainable",
    "count_parameters",
    "merge_lora",
    "lora_state_dict",
    "save_lora",
    "load_lora",
]
