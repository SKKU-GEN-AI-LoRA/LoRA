# LoRA from Scratch

A from-scratch implementation of **LoRA (Low-Rank Adaptation)** — no `peft`, no
Hugging Face `Trainer` — and a systematic study of how **rank**, **target modules**,
and **scaling** affect adapting a small LLM to the medical-QA domain, measured on
standard benchmarks for both in-domain and transfer performance.

- **Base model:** Qwen2.5-1.5B-Instruct (fallback: 0.5B-Instruct)
- **Train:** `openlifescienceai/medmcqa`
- **Evaluate:** MedMCQA (in-domain) · MedQA / PubMedQA (transfer) — MCQ accuracy via
  per-choice loglikelihood
- The LoRA layer, module injection, weight merging, training loop, and sweep harness
  are all implemented from scratch.

## Install

```bash
uv sync
```

## Quick check (no network/GPU)

```bash
pytest tests/ -q   # LoRA math, injection, merge, data adapters, end-to-end smoke
```

## Reproduce (single notebook)

Everything — concept, from-scratch implementation, training/eval, the
rank/module/scaling/method sweeps, results, and a qualitative generation demo — lives
in **`notebooks/LoRA_medical_QA.ipynb`**. A `SMOKE` flag at the top selects the scale:

- `SMOKE = True` (default) → `configs/smoke/*` (0.5B, small subsets): runs on CPU/MPS in minutes
- `SMOKE = False` → `configs/*.yaml` (1.5B, full): GPU recommended

## Experiments (CLI)

```bash
# baseline (r=8, {q,v})
python scripts/train.py --config configs/lora_r8_qv.yaml          # full
python scripts/train.py --config configs/smoke/lora_r8_qv.yaml    # smoke

# one-factor-at-a-time sweeps
python eval/sweep.py --sweep configs/sweep_rank.yaml      # rank
python eval/sweep.py --sweep configs/sweep_module.yaml    # target modules
python eval/sweep.py --sweep configs/sweep_alpha.yaml     # scaling alpha/r
python eval/sweep.py --sweep configs/sweep_method.yaml    # LoRA vs Full FT vs Zero-shot

# re-evaluate a checkpoint / qualitative generation
python scripts/eval.py --config configs/lora_r8_qv.yaml --ckpt checkpoints/lora_r8_qv/lora.pt
python scripts/generate.py --config configs/lora_r8_qv.yaml \
    --ckpt checkpoints/lora_r8_qv/lora.pt --question "..."
```

Each run writes `output_dir/results.json` (accuracy, trainable %, peak memory, time);
each sweep aggregates to `output_root/results.{json,csv}`.

## Structure

```
config.py            # experiment config (YAML -> dataclass)
lora/                # from-scratch core
  layer.py           #   LoRALinear (W = W0 + (alpha/r)*BA)
  inject.py          #   inject_lora: traverse + replace modules, count params
  merge.py           #   merge + LoRA-only save/load
data/                # medical QA loading + unified schema + MCQ formatting/tokenization
eval/                # metrics (MCQ acc / perplexity / memory) + sweep harness
scripts/             # train · eval · generate (no HF Trainer; explicit loop)
configs/             # experiment + sweep YAML
notebooks/           # LoRA_medical_QA.ipynb — end-to-end
tests/               # offline core unit tests
```

## From-scratch principle

No `peft` and no Hugging Face `Trainer`. The LoRA layer, injection, merge, training
loop, and sweep harness are all written directly. Implementation correctness is
checked by pure-torch unit tests (`tests/test_lora.py`: formula match, B=0 invariance,
merge invariance).

## Datasets

| Dataset | Source | Split | Use |
|---|---|---|---|
| MedMCQA | `openlifescienceai/medmcqa` | validation | in-domain (also training) |
| MedQA | `openlifescienceai/medqa` | test | transfer |
| PubMedQA | `openlifescienceai/pubmedqa` | test | transfer |

Inspect the real schemas with `python scripts/verify_datasets.py`.
