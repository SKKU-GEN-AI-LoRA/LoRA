"""Training CLI — no HF Trainer, hand-written training loop (design §5, §8.1).

    python scripts/train.py --config configs/lora_r8_qv.yaml
    python scripts/train.py --config configs/lora_r8_qv.yaml --r 16 --output-dir checkpoints/r16

With method=lora, inject LoRA and train only LoRA params; with full, train everything;
with zeroshot, skip training and only evaluate. Results (accuracy, trainable%, peak
memory, time) are saved to output_dir/results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from config import Config, load_config
from data.format import encode_train_example
from data.load import load_qa_dataset
from eval.metrics import evaluate_accuracy, peak_memory_gb, reset_peak_memory
from lora import (
    count_parameters,
    inject_lora,
    mark_only_lora_as_trainable,
    save_lora,
)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(cfg: Config, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=_DTYPES[cfg.dtype]
    )
    model.to(device)
    return model, tokenizer


def _collate(batch: list[dict], pad_id: int):
    """Right padding + labels padded with -100 (ignored in causal LM loss)."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        ids, lab = b["input_ids"], b["labels"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def _setup_method(model: torch.nn.Module, cfg: Config) -> list[str]:
    """Set trainable params per method. Returns replaced module paths (lora)."""
    if cfg.method == "lora":
        _, replaced = inject_lora(
            model,
            cfg.target_modules,
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.lora_dropout,
            init=cfg.lora_init,
        )
        mark_only_lora_as_trainable(model)
        return replaced
    if cfg.method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return []
    if cfg.method == "zeroshot":
        for p in model.parameters():
            p.requires_grad_(False)
        return []
    raise ValueError(f"unknown method: {cfg.method} (lora|full|zeroshot)")


def _train_loop(model, tokenizer, cfg: Config, device) -> dict:
    train_examples = load_qa_dataset(
        cfg.train_dataset, "train", cfg.train_subset, cfg.seed
    )
    encoded = [encode_train_example(ex, tokenizer, cfg.max_length) for ex in train_examples]
    loader = DataLoader(
        encoded,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )

    updates_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_updates = cfg.max_steps or updates_per_epoch * cfg.epochs
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * cfg.warmup_ratio), total_updates
    )

    reset_peak_memory()
    model.train()
    losses: list[float] = []
    step = 0
    start = time.time()
    stop = False
    for _epoch in range(cfg.epochs):
        optimizer.zero_grad()
        for i, (ids, labels, attn) in enumerate(loader):
            ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            (loss / cfg.grad_accum).backward()
            losses.append(loss.item())
            if (i + 1) % cfg.grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % cfg.log_every == 0:
                    recent = sum(losses[-cfg.log_every:]) / min(len(losses), cfg.log_every)
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  step {step}/{total_updates}  loss={recent:.4f}  lr={lr_now:.2e}")
                if cfg.max_steps and step >= cfg.max_steps:
                    stop = True
                    break
        if stop:
            break

    return {
        "train_time_sec": round(time.time() - start, 1),
        "peak_memory_gb": round(peak_memory_gb(), 3),
        "final_loss": round(sum(losses[-50:]) / max(len(losses[-50:]), 1), 4),
        "loss_curve": [round(x, 4) for x in losses],
        "updates": step,
    }


def run_experiment(cfg: Config) -> dict:
    """Run one experiment (= one Config) end to end: setup -> train -> eval -> save results."""
    set_seed(cfg.seed)
    device = get_device()
    print(f"[{cfg.method}] {cfg.model_name} on {device} "
          f"(r={cfg.r}, alpha={cfg.alpha}, targets={cfg.target_modules})")

    model, tokenizer = load_model_and_tokenizer(cfg, device)
    _setup_method(model, cfg)
    trainable, total = count_parameters(model)
    print(f"  trainable {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    if cfg.grad_checkpointing and cfg.method != "zeroshot":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    results: dict = {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
    }

    outdir = pathlib.Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.method != "zeroshot":
        results.update(_train_loop(model, tokenizer, cfg, device))
        if cfg.method == "lora":
            save_lora(model, outdir / "lora.pt")
        else:
            model.save_pretrained(outdir / "full")
            tokenizer.save_pretrained(outdir / "full")

    if cfg.grad_checkpointing:
        model.config.use_cache = True

    eval_results: dict = {}
    for name in cfg.eval_datasets:
        examples = load_qa_dataset(name, "eval", cfg.eval_subset, cfg.seed)
        res = evaluate_accuracy(model, tokenizer, examples, name, cfg.max_length, device)
        print(f"  {res}")
        eval_results[name] = {
            "accuracy": res.accuracy,
            "n": res.n,
            "correct": res.correct,
            "pred_hist": res.per_choice_hist,
        }
    results["eval"] = eval_results

    (outdir / "results.json").write_text(
        json.dumps({"config": cfg.to_dict(), "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  → saved {outdir / 'results.json'}")
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA medical QA training (hand-written loop)")
    p.add_argument("--config", required=True, help="path to experiment config YAML")
    # overrides for sweep convenience (override config values when given)
    p.add_argument("--method", choices=["lora", "full", "zeroshot"])
    p.add_argument("--r", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--target-modules", nargs="+", dest="target_modules")
    p.add_argument("--train-subset", type=int, dest="train_subset")
    p.add_argument("--max-steps", type=int, dest="max_steps")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--model-name", dest="model_name")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg = load_config(args.config, **overrides)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
