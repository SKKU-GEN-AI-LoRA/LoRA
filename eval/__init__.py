"""Evaluation metrics + sweep harness (design §4.3, §6.2)."""

from .metrics import (
    EvalResult,
    evaluate_accuracy,
    peak_memory_gb,
    perplexity,
    reset_peak_memory,
)

__all__ = [
    "EvalResult",
    "evaluate_accuracy",
    "perplexity",
    "peak_memory_gb",
    "reset_peak_memory",
]
