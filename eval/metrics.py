"""Evaluation metrics: MCQ accuracy (per-choice loglikelihood), perplexity, memory (design §6.2).

MCQ accuracy is the primary metric. We compute the loglikelihood of each choice
letter candidate and take the argmax as the prediction — scored with a single forward
pass without generation, so it is fast and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from data.format import encode_eval_candidates
from data.load import Example


@dataclass
class EvalResult:
    dataset: str
    accuracy: float
    n: int
    correct: int
    per_choice_hist: dict[int, int] = field(default_factory=dict)  # pred dist (collapse diag)

    def __str__(self) -> str:
        return f"{self.dataset}: acc={self.accuracy:.4f} ({self.correct}/{self.n})"


@torch.no_grad()
def _candidate_loglikelihood(
    model: torch.nn.Module,
    input_ids: list[int],
    target_start: int,
    target_ids: list[int],
    device: torch.device,
) -> float:
    """Sum of logprobs of target_ids starting from target_start position in input_ids."""
    ids = torch.tensor([input_ids], device=device)
    logits = model(ids).logits  # (1, T, V)
    # The logit predicting the token at position t lives at position t-1.
    logprobs = F.log_softmax(logits[0].float(), dim=-1)
    total = 0.0
    for offset, tok in enumerate(target_ids):
        pos = target_start + offset
        total += logprobs[pos - 1, tok].item()
    return total


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    dataset_name: str,
    max_length: int = 512,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> EvalResult:
    """Compute MCQ accuracy via argmax of per-choice loglikelihood."""
    device = device or next(model.parameters()).device
    model.eval()
    correct = 0
    hist: dict[int, int] = {}
    iterator = tqdm(examples, desc=f"eval/{dataset_name}", disable=not show_progress)
    for ex in iterator:
        cands = encode_eval_candidates(ex, tokenizer, max_length)
        lls = [
            _candidate_loglikelihood(
                model, c["input_ids"], c["target_start"], c["target_ids"], device
            )
            for c in cands
        ]
        pred = max(range(len(lls)), key=lambda i: lls[i])
        hist[pred] = hist.get(pred, 0) + 1
        correct += int(pred == ex.answer_idx)
    n = len(examples)
    return EvalResult(dataset_name, correct / n, n, correct, hist)


@torch.no_grad()
def perplexity(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = 512,
    device: torch.device | None = None,
) -> float:
    """Held-out perplexity (auxiliary metric, design §6.2). Token-weighted mean NLL -> exp."""
    device = device or next(model.parameters()).device
    model.eval()
    total_nll, total_tok = 0.0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = {k: v.to(device) for k, v in ids.items()}
        out = model(**ids, labels=ids["input_ids"])
        n_tok = ids["input_ids"].numel() - 1
        total_nll += out.loss.item() * n_tok
        total_tok += n_tok
    return float(torch.exp(torch.tensor(total_nll / max(total_tok, 1))))


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    """torch.cuda.max_memory_allocated -> GB (design §6.2 efficiency metric)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0
