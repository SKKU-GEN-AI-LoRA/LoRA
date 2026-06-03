"""End-to-end smoke test on a tiny locally-built Qwen2 — no network, no GPU.

Exercises the real pipeline functions together: inject → trainable mask →
encode → collate → forward/backward/step → loglikelihood eval → save/load →
merge. Proves the moving parts wire up correctly before any expensive run.

    pytest tests/test_smoke_e2e.py -q
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from data.format import encode_train_example
from data.load import Example
from eval.metrics import evaluate_accuracy
from lora import (
    count_parameters,
    inject_lora,
    load_lora,
    lora_state_dict,
    mark_only_lora_as_trainable,
    merge_lora,
    save_lora,
)
from scripts.train import _collate

VOCAB = 256


class TinyTokenizer:
    """Deterministic word→id tokenizer bounded to the model vocab."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def _tid(self, w: str) -> int:
        return 1 + (hash(w) % (VOCAB - 1))  # 0 reserved for padding

    def __call__(self, text: str, add_special_tokens: bool = True, **_):
        toks = (["<bos>"] if add_special_tokens else []) + text.split()
        return {"input_ids": [self._tid(t) for t in toks]}


def _tiny_model() -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg)


def _examples(n: int = 8) -> list[Example]:
    return [
        Example(prompt=f"symptom {i} differential", choices=["alpha", "beta", "gamma", "delta"],
                answer_idx=i % 4)
        for i in range(n)
    ]


def test_end_to_end_pipeline(tmp_path):
    tok = TinyTokenizer()
    model = _tiny_model()

    # inject + freeze base
    _, replaced = inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=8.0)
    assert len(replaced) == 2 * model.config.num_hidden_layers
    mark_only_lora_as_trainable(model)
    trainable, total = count_parameters(model)
    assert 0 < trainable < total

    # one real train step on the masked-loss batch
    examples = _examples()
    encoded = [encode_train_example(ex, tok, max_length=64) for ex in examples]
    ids, labels, attn = _collate(encoded, tok.pad_token_id)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    model.train()
    before = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
    loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    # only LoRA params moved; frozen base untouched
    moved = [n for n, p in model.named_parameters()
             if p.requires_grad and not torch.equal(p, before[n])]
    assert moved, "no trainable parameter updated"
    frozen = next(p for n, p in model.named_parameters() if "lora_" not in n)
    assert frozen.grad is None

    # eval runs and predicts a valid choice (accuracy in [0,1])
    res = evaluate_accuracy(model, tok, examples, "tiny", max_length=64,
                            device=torch.device("cpu"), show_progress=False)
    assert res.n == len(examples)
    assert 0.0 <= res.accuracy <= 1.0
    assert set(res.per_choice_hist).issubset(set(range(4)))

    # save/load roundtrip is exact
    ckpt = tmp_path / "lora.pt"
    save_lora(model, ckpt)
    saved = lora_state_dict(model)
    fresh = _tiny_model()
    inject_lora(fresh, ["q_proj", "v_proj"], r=4, alpha=8.0)
    load_lora(fresh, ckpt)
    for name, tensor in lora_state_dict(fresh).items():
        assert torch.equal(tensor, saved[name])


def test_merge_matches_unmerged_logits():
    """Merging ΔW into W0 leaves forward output unchanged (zero inference cost)."""
    tok = TinyTokenizer()
    model = _tiny_model()
    inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=8.0)
    # give B a non-trivial value so ΔW != 0
    for m in model.modules():
        if hasattr(m, "lora_B"):
            torch.nn.init.normal_(m.lora_B, std=0.05)
    ids = torch.tensor([tok("a tiny medical prompt")["input_ids"]])
    model.eval()
    with torch.no_grad():
        before = model(ids).logits.clone()
        merge_lora(model)
        after = model(ids).logits
    assert torch.allclose(before, after, atol=1e-4)
