"""Mock-based unit tests for the data/load.py adapters + data/format.py formatting.

Only verifies the parts that work without HuggingFace datasets/transformers.
Actual dataset schema measurement is handled by scripts/verify_datasets.py.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.format import LETTERS, build_prompt, encode_eval_candidates, encode_train_example
from data.load import Example, _adapt_medmcqa, _adapt_medqa, _adapt_pubmedqa

# ── Adapters: per-dataset raw record → unified schema ────────────────────

def test_adapt_medmcqa_basic():
    ex = _adapt_medmcqa(
        {"question": "Q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 2}
    )
    assert ex is not None
    assert ex.prompt == "Q?" and ex.choices == ["a", "b", "c", "d"]
    assert ex.answer_idx == 2


def test_adapt_medmcqa_drops_test_dummy_label():
    """MedMCQA test split has hidden answers (cop=-1) → returns None (R8)."""
    ex = _adapt_medmcqa(
        {"question": "Q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": -1}
    )
    assert ex is None


def test_adapt_medqa_nested_data_letter_answer():
    rec = {
        "data": {
            "Question": "Q?",
            "Options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "Correct Option": "C",
        }
    }
    ex = _adapt_medqa(rec)
    assert ex is not None
    assert ex.prompt == "Q?" and ex.choices == ["a", "b", "c", "d"]
    assert ex.answer_idx == 2  # 'C' → index 2


def test_adapt_pubmedqa_three_choices():
    rec = {
        "data": {
            "Question": "Is X true?",
            "Final Decision": "yes",
            "Context": ["passage 1", "passage 2"],
        }
    }
    ex = _adapt_pubmedqa(rec)
    assert ex is not None
    assert ex.choices == ["yes", "no", "maybe"]
    assert ex.answer_idx == 0
    assert ex.context and "passage 1" in ex.context


def test_adapt_pubmedqa_invalid_label_dropped():
    assert _adapt_pubmedqa({"data": {"Question": "Q?", "Final Decision": "??"}}) is None


def test_adapt_pubmedqa_options_letter_schema():
    """Measured schema: Options{A,B,C} + 'Correct Option' letter (isomorphic to MedQA)."""
    rec = {
        "data": {
            "Question": "Is X valuable?",
            "Options": {"A": "yes", "B": "no", "C": "maybe"},
            "Correct Option": "A",
            "Context": ["passage one", "passage two"],
        }
    }
    ex = _adapt_pubmedqa(rec)
    assert ex is not None
    assert ex.choices == ["yes", "no", "maybe"]
    assert ex.answer_idx == 0
    assert ex.context and "passage one" in ex.context


# ── Formatter: prompt construction ───────────────────────────────────────

def test_build_prompt_contains_letters():
    ex = Example(prompt="Q?", choices=["alpha", "beta", "gamma"], answer_idx=1)
    p = build_prompt(ex)
    for i in range(3):
        assert f"{LETTERS[i]}." in p
    assert p.endswith("Answer:")
    assert "Q?" in p


def test_build_prompt_includes_context():
    ex = Example(prompt="Q?", choices=["yes", "no"], answer_idx=0, context="CTX TEXT")
    assert "CTX TEXT" in build_prompt(ex)


# ── Tokenize: verify loss masking / target position with a mock tokenizer ──

class MockTokenizer:
    """Minimal tokenizer assuming 1 word = 1 token."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _tid(self, w: str) -> int:
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab) + 1  # 0 is unused
        return self.vocab[w]

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs):
        toks = text.split()
        if add_special_tokens:
            toks = ["<bos>"] + toks
        return {"input_ids": [self._tid(t) for t in toks]}


def test_encode_train_masks_prompt_keeps_answer_label():
    tok = MockTokenizer()
    ex = Example(prompt="Q?", choices=["a", "b", "c"], answer_idx=1)
    out = encode_train_example(ex, tok, max_length=256)
    ids, labels = out["input_ids"], out["labels"]
    assert len(ids) == len(labels)
    # last token = answer letter ' B'. only the last position of labels is unmasked.
    assert labels[-1] == ids[-1]
    assert all(lbl == -100 for lbl in labels[:-1])


def test_encode_eval_candidates_one_per_choice():
    tok = MockTokenizer()
    ex = Example(prompt="Q?", choices=["a", "b", "c", "d"], answer_idx=0)
    cands = encode_eval_candidates(ex, tok, max_length=256)
    assert len(cands) == 4
    # each candidate is the same prompt length + 1 letter token.
    lens = {len(c["input_ids"]) for c in cands}
    assert len(lens) == 1
    for c in cands:
        assert len(c["target_ids"]) >= 1
        assert c["target_start"] + len(c["target_ids"]) == len(c["input_ids"])


def test_eval_candidates_long_prompt_keeps_target_in_bounds():
    """Even when a long prompt (MedQA passage) exceeds max_length, the answer token
    must be preserved and target_start must point inside the sequence
    (loglikelihood IndexError regression)."""
    tok = MockTokenizer()
    long_q = " ".join(f"word{i}" for i in range(500))  # prompt alone exceeds max_length
    ex = Example(prompt=long_q, choices=["a", "b", "c", "d"], answer_idx=2)
    max_length = 64
    for c in encode_eval_candidates(ex, tok, max_length=max_length):
        assert len(c["input_ids"]) <= max_length
        # target_start..end are all inside the sequence.
        assert 0 <= c["target_start"] < len(c["input_ids"])
        assert c["target_start"] + len(c["target_ids"]) == len(c["input_ids"])
        # the answer token actually remains at the end of the sequence.
        assert c["input_ids"][c["target_start"]:] == c["target_ids"]


def test_encode_train_long_prompt_preserves_answer_label():
    """Even with a long prompt, the answer letter label must survive (avoid all -100)."""
    tok = MockTokenizer()
    long_q = " ".join(f"word{i}" for i in range(500))
    ex = Example(prompt=long_q, choices=["a", "b", "c", "d"], answer_idx=1)
    out = encode_train_example(ex, tok, max_length=64)
    assert len(out["input_ids"]) <= 64
    assert len(out["input_ids"]) == len(out["labels"])
    assert out["labels"][-1] == out["input_ids"][-1]  # loss on the answer token
    assert any(lbl != -100 for lbl in out["labels"])
