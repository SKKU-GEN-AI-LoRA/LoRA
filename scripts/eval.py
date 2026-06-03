"""Re-measure benchmark accuracy from a saved checkpoint (LoRA-only or full) (design §6.2).

    python scripts/eval.py --config configs/lora_r8_qv.yaml --ckpt checkpoints/run/lora.pt
    python scripts/eval.py --config configs/full_ft.yaml --ckpt checkpoints/run/full
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config
from data.load import load_qa_dataset
from eval.metrics import evaluate_accuracy
from lora import inject_lora, load_lora
from scripts.train import _DTYPES, get_device


def main() -> None:
    p = argparse.ArgumentParser(description="checkpoint evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", help="LoRA .pt or full directory (omit for baseline=untrained)")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    if cfg.method == "full" and args.ckpt:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=_DTYPES[cfg.dtype])
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=_DTYPES[cfg.dtype])
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if cfg.method == "lora" and args.ckpt:
            inject_lora(model, cfg.target_modules, r=cfg.r, alpha=cfg.alpha, init=cfg.lora_init)
            load_lora(model, args.ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    out: dict = {}
    for name in cfg.eval_datasets:
        examples = load_qa_dataset(name, "eval", cfg.eval_subset, cfg.seed)
        res = evaluate_accuracy(model, tokenizer, examples, name, cfg.max_length, device)
        print(res)
        out[name] = {"accuracy": res.accuracy, "n": res.n, "correct": res.correct}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
