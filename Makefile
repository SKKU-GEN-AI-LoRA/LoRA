# LoRA from scratch — shortcut commands for reproducing experiments.
.PHONY: help install test verify e1 sweeps rank module alpha method clean

PY = python

help:
	@echo "Targets:"
	@echo "  install    - set up the uv environment"
	@echo "  test       - offline core tests (no network/GPU needed)"
	@echo "  verify     - inspect the real dataset schemas (network)"
	@echo "  e1         - baseline (r=8, qv)"
	@echo "  rank/module/alpha/method - individual sweeps"
	@echo "  sweeps     - run all four sweeps"

install:
	uv sync

test:
	$(PY) -m pytest tests/ -q

verify:
	$(PY) scripts/verify_datasets.py

e1:
	$(PY) scripts/train.py --config configs/lora_r8_qv.yaml

rank:
	$(PY) eval/sweep.py --sweep configs/sweep_rank.yaml
module:
	$(PY) eval/sweep.py --sweep configs/sweep_module.yaml
alpha:
	$(PY) eval/sweep.py --sweep configs/sweep_alpha.yaml
method:
	$(PY) eval/sweep.py --sweep configs/sweep_method.yaml

sweeps: rank module alpha method

clean:
	rm -rf checkpoints/* __pycache__ */__pycache__
