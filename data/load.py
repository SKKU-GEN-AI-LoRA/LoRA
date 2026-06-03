"""Medical QA dataset loading + per-dataset adapters (design §6.3).

The three datasets have different structures, so adapters normalize them into an
internal unified schema ``Example{prompt, choices, answer_idx}``. The number of
choices may differ per dataset (4-way / 3-way) -> eval works regardless of count.

The schemas were verified empirically with scripts/verify_datasets.py. Adapters try
the documented field names first and defend against variants (key case / nested data).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Train/eval split mapping (design §6.3 table).
#   medmcqa : test answers are private -> score on validation (R8)
#   medqa   : test (USMLE 4-way multiple choice)
#   pubmedqa: test (yes/no/maybe 3-way multiple choice)
_DATASET_SPEC: dict[str, dict[str, str]] = {
    "medmcqa": {"path": "openlifescienceai/medmcqa", "train": "train", "eval": "validation"},
    "medqa": {"path": "openlifescienceai/medqa", "train": "train", "eval": "test"},
    "pubmedqa": {"path": "openlifescienceai/pubmedqa", "train": "train", "eval": "test"},
}


@dataclass
class Example:
    """Unified schema. Loss/scoring depend only on answer_idx (correct choice index)."""

    prompt: str
    choices: list[str]
    answer_idx: int
    context: str | None = None  # When a passage is present (e.g. PubMedQA)


def _row(record: dict[str, Any]) -> dict[str, Any]:
    """openlifescienceai datasets often nest the actual fields under ``data``."""
    inner = record.get("data")
    return inner if isinstance(inner, dict) else record


def _adapt_medmcqa(record: dict[str, Any]) -> Example | None:
    r = _row(record)
    cop = r.get("cop")
    if cop is None or int(cop) < 0:  # guard against dummy labels in the test split
        return None
    return Example(
        prompt=r["question"].strip(),
        choices=[str(r["opa"]), str(r["opb"]), str(r["opc"]), str(r["opd"])],
        answer_idx=int(cop),
    )


def _adapt_medqa(record: dict[str, Any]) -> Example | None:
    r = _row(record)
    options = r.get("Options") or r.get("options")
    if isinstance(options, dict):
        letters = sorted(options)  # 'A','B','C','D'
        choices = [str(options[k]) for k in letters]
        correct = str(r.get("Correct Option") or r.get("answer_idx") or r["answer"]).strip()
        answer_idx = letters.index(correct) if correct in letters else int(correct)
    else:  # variant where choices are a list
        choices = [str(c) for c in options]
        answer_idx = int(r["answer_idx"])
    return Example(prompt=r["Question"].strip() if "Question" in r else r["question"].strip(),
                   choices=choices, answer_idx=answer_idx)


def _adapt_pubmedqa(record: dict[str, Any]) -> Example | None:
    r = _row(record)
    question = (r.get("Question") or r.get("QUESTION") or r.get("question") or "").strip()
    if not question:
        return None

    contexts = r.get("Context") or r.get("CONTEXTS") or r.get("context")
    if isinstance(contexts, dict):
        contexts = contexts.get("contexts") or contexts.get("CONTEXTS")
    context = " ".join(contexts) if isinstance(contexts, list) else (contexts or None)

    # Observed schema (openlifescienceai/pubmedqa): same as MedQA, i.e.
    # Options{A,B,C} + 'Correct Option' (letter). Derive the answer index from the letter.
    options = r.get("Options") or r.get("options")
    if isinstance(options, dict):
        letters = sorted(options)  # 'A','B','C'
        choices = [str(options[k]) for k in letters]
        correct = str(r.get("Correct Option") or r.get("answer") or "").strip()
        if correct not in letters:
            return None
        return Example(prompt=question, choices=choices,
                       answer_idx=letters.index(correct), context=context)

    # Legacy schema fallback: Final Decision / Correct Answer = yes/no/maybe
    decision = str(
        r.get("Final Decision") or r.get("final_decision")
        or r.get("Correct Answer") or r.get("answer")
    ).strip().lower()
    choices = ["yes", "no", "maybe"]
    if decision not in choices:
        return None
    return Example(prompt=question, choices=choices, answer_idx=choices.index(decision),
                   context=context)


_ADAPTERS: dict[str, Callable[[dict[str, Any]], Example | None]] = {
    "medmcqa": _adapt_medmcqa,
    "medqa": _adapt_medqa,
    "pubmedqa": _adapt_pubmedqa,
}


def load_qa_dataset(
    name: str, split: str = "eval", subset: int | None = None, seed: int = 42
) -> list[Example]:
    """Load the ``name`` dataset as a list of unified-schema Examples.

    Args:
        name : medmcqa | medqa | pubmedqa
        split: "train" or "eval" (mapped to the actual HF split)
        subset: first N examples (None=all). For train, take N after a seeded shuffle.
    """
    if name not in _DATASET_SPEC:
        raise ValueError(f"Unknown dataset: {name} (medmcqa|medqa|pubmedqa)")
    # HF datasets are only needed at train/eval runtime. Adapter unit tests run without it.
    from datasets import load_dataset

    spec = _DATASET_SPEC[name]
    hf_split = spec[split]
    ds = load_dataset(spec["path"], split=hf_split)
    if split == "train" and subset is not None:
        ds = ds.shuffle(seed=seed)

    adapt = _ADAPTERS[name]
    examples: list[Example] = []
    for record in ds:
        ex = adapt(record)
        if ex is not None and 0 <= ex.answer_idx < len(ex.choices):
            examples.append(ex)
        if subset is not None and len(examples) >= subset:
            break
    if not examples:
        raise RuntimeError(f"No valid examples from {name}/{hf_split} — check the schema")
    return examples
