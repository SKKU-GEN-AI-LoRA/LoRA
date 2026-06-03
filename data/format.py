"""MCQ instruction formatting + tokenization (design §6.3, §8.1).

Strategy: present every choice as a letter (A, B, ...) and output the answer as a
single letter.
  - Train: only the letter token in prompt + " {letter}" gets loss (rest masked -100).
  - Eval: compare each letter candidate's loglikelihood -> argmax (any number of choices).
Letter candidates are all single tokens, so comparing LL without length normalization
is fair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # transformers only needed at train/eval runtime. Offline tests use a mock.
    from transformers import PreTrainedTokenizerBase

from .load import Example

LETTERS = "ABCDEFGH"

_INSTRUCTION = (
    "The following is a multiple choice question about medicine. "
    "Answer with the letter of the correct option."
)


def build_prompt(ex: Example) -> str:
    """Build the prompt string up to just before the answer (letter not included)."""
    lines = [_INSTRUCTION, ""]
    if ex.context:
        lines += [f"Context: {ex.context}", ""]
    lines.append(f"Question: {ex.prompt}")
    for i, choice in enumerate(ex.choices):
        lines.append(f"{LETTERS[i]}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def _letter_token_ids(tokenizer: PreTrainedTokenizerBase, n_choices: int) -> list[list[int]]:
    """Token id sequences for ' A', ' B', ... (usually a single token each)."""
    return [
        tokenizer(f" {LETTERS[i]}", add_special_tokens=False)["input_ids"]
        for i in range(n_choices)
    ]


def _fit_prompt(prompt_ids: list[int], cont_len: int, max_length: int) -> list[int]:
    """If prompt + continuation exceeds max_length, truncate the prompt **from the left**.

    Long passages like MedQA (USMLE) can exceed max_length with the prompt alone. A naive
    right truncation would cut off the answer tokens, making target_start point outside the
    sequence. Since the choices / "Answer:" are at the end of the prompt, we drop the front
    (the instruction) and keep the tail so the answer continuation is always included.
    """
    keep = max(max_length - cont_len, 1)
    return prompt_ids[-keep:] if len(prompt_ids) > keep else prompt_ids


def encode_train_example(
    ex: Example, tokenizer: PreTrainedTokenizerBase, max_length: int = 512
) -> dict[str, list[int]]:
    """Training encoding: loss (labels) only on the correct letter token, rest -100."""
    prompt = build_prompt(ex)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(f" {LETTERS[ex.answer_idx]}", add_special_tokens=False)["input_ids"]

    prompt_ids = _fit_prompt(prompt_ids, len(answer_ids), max_length)
    input_ids = prompt_ids + list(answer_ids)
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return {"input_ids": input_ids, "labels": labels}


def encode_eval_candidates(
    ex: Example, tokenizer: PreTrainedTokenizerBase, max_length: int = 512
) -> list[dict[str, list[int]]]:
    """Eval: build (input_ids, target token position/target) for each letter candidate.

    Returns, for each choice i, dict{input_ids, target_ids, target_start}:
        sum of logprobs from target_start = that choice's loglikelihood.
    """
    prompt = build_prompt(ex)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    candidates = []
    for letter_ids in _letter_token_ids(tokenizer, len(ex.choices)):
        fitted = _fit_prompt(prompt_ids, len(letter_ids), max_length)
        candidates.append(
            {
                "input_ids": fitted + list(letter_ids),
                "target_ids": list(letter_ids),
                "target_start": len(fitted),
            }
        )
    return candidates
