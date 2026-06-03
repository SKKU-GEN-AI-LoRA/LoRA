"""Medical QA dataset schema verification script.

Run on Colab/locally to confirm that the schema recorded in design.md §6.3 matches reality.
In particular, to visually confirm PubMedQA's nested `data` sub-key (the answer field name).

    pip install datasets
    python scripts/verify_datasets.py
"""

from __future__ import annotations

import json

from datasets import load_dataset

DATASETS = {
    "medmcqa": ("openlifescienceai/medmcqa", ["train", "validation", "test"]),
    "medqa": ("openlifescienceai/medqa", ["train", "test", "dev"]),
    "pubmedqa": ("openlifescienceai/pubmedqa", ["train", "validation", "test"]),
}


def _peek(name: str, path: str, splits: list[str]) -> None:
    print(f"\n{'=' * 70}\n{name}  ({path})\n{'=' * 70}")
    for split in splits:
        try:
            ds = load_dataset(path, split=split)
        except Exception as exc:  # the split name may differ
            print(f"  [{split}] load failed: {exc}")
            continue
        print(f"  [{split}] rows={len(ds):,}  columns={ds.column_names}")
        if split == splits[0]:
            print("  first row sample:")
            print(json.dumps(ds[0], ensure_ascii=False, indent=2)[:1500])


def main() -> None:
    for name, (path, splits) in DATASETS.items():
        _peek(name, path, splits)
    print(
        "\nCheck points:\n"
        "  - medmcqa: cop(int 0-3), opa~opd, whether test's cop is a dummy\n"
        "  - medqa:   data.Question / data.Options / data.'Correct Option'\n"
        "  - pubmedqa: answer field under data (e.g. Final Decision) and label (yes/no/maybe)\n"
    )


if __name__ == "__main__":
    main()
