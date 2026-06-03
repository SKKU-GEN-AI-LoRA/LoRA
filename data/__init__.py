"""Medical QA data pipeline: load -> normalize to unified schema -> MCQ format -> tokenize."""

from .format import (
    LETTERS,
    build_prompt,
    encode_eval_candidates,
    encode_train_example,
)
from .load import Example, load_qa_dataset

__all__ = [
    "Example",
    "load_qa_dataset",
    "LETTERS",
    "build_prompt",
    "encode_train_example",
    "encode_eval_candidates",
]
