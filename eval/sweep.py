"""(rank × module × alpha) sweep harness (design §4.3, §7 E2–E5).

sweep YAML structure:
    base: configs/lora_r8_qv.yaml      # base config
    output_root: checkpoints/sweeps/rank
    runs:                               # each run = override applied on top of base
      - {name: r1,  r: 1}
      - {name: r8,  r: 8}
      - {name: r32, r: 32}

Following the R4 mitigation, designed as one-factor-at-a-time (one axis at a time).
Collects each run's results and saves them to the sweep directory as results.json / results.csv.

    python eval/sweep.py --sweep configs/lora_sweep.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from config import Config
from scripts.train import run_experiment


def _load_base(path: str) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}


def run_sweep(sweep_path: str) -> list[dict]:
    spec = yaml.safe_load(pathlib.Path(sweep_path).read_text(encoding="utf-8"))
    base = _load_base(spec["base"]) if "base" in spec else {}
    output_root = pathlib.Path(spec.get("output_root", "checkpoints/sweeps/run"))
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    runs = spec["runs"]
    for i, override in enumerate(runs, 1):
        override = dict(override)
        name = override.pop("name", f"run{i}")
        merged = {**base, **override}
        merged["output_dir"] = str(output_root / name)
        cfg = Config(**merged)
        print(f"\n{'=' * 70}\n[sweep {i}/{len(runs)}] {name}\n{'=' * 70}")
        res = run_experiment(cfg)

        row = {
            "name": name,
            "method": cfg.method,
            "r": cfg.r,
            "alpha": cfg.alpha,
            "scaling": round(cfg.scaling, 3),
            "target_modules": "+".join(cfg.target_modules),
            "trainable_params": res["trainable_params"],
            "trainable_pct": res["trainable_pct"],
            "peak_memory_gb": res.get("peak_memory_gb"),
            "train_time_sec": res.get("train_time_sec"),
            "final_loss": res.get("final_loss"),
        }
        for ds, m in res["eval"].items():
            row[f"acc_{ds}"] = round(m["accuracy"], 4)
        rows.append(row)

    (output_root / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        fields = sorted({k for r in rows for k in r})
        with (output_root / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n→ sweep results saved: {output_root}/results.{{json,csv}}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA sweep harness")
    p.add_argument("--sweep", required=True, help="sweep definition YAML")
    args = p.parse_args()
    run_sweep(args.sweep)


if __name__ == "__main__":
    main()
