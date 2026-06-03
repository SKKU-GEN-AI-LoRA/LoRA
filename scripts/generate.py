"""Generate responses with a trained model — qualitative comparison demo (design §7.1).

Visually compare the responses and rationales of zero-shot vs LoRA vs Full FT on the same item.

    python scripts/generate.py --config configs/lora_r8_qv.yaml --ckpt checkpoints/run/lora.pt \
        --question "A 24-year-old patient presents with ..."
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config
from lora import inject_lora, load_lora, merge_lora
from scripts.train import _DTYPES, get_device


def main() -> None:
    p = argparse.ArgumentParser(description="response generation demo")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", help="LoRA .pt (omit for zero-shot)")
    p.add_argument("--question", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=_DTYPES[cfg.dtype])
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.ckpt:
        inject_lora(model, cfg.target_modules, r=cfg.r, alpha=cfg.alpha, init=cfg.lora_init)
        load_lora(model, args.ckpt)
        merge_lora(model)  # zero inference latency (design §8.2)
    model.to(device).train(False)

    messages = [{"role": "user", "content": args.question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    label = "LoRA" if args.ckpt else "zero-shot"
    print(f"\n=== {label} response ===\n{text}")


if __name__ == "__main__":
    main()
