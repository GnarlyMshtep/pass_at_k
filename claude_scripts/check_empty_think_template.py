"""
Dummy-0 sanity check: load Qwen3-8B tokenizer, override its chat template with
OpenPipe/Qwen3-14B-Instruct's `chat_template.jinja` (which appends empty <think></think>
tags as the generation prompt), verify:
  1. The rendered prompt ends with the expected empty-think suffix.
  2. encode→decode is a round-trip for that prompt.
  3. Vocab is unchanged vs. stock Qwen3-8B tokenizer (same size, same special ids).

No GPU / model weights needed. Fast fail before touching dummy_verl.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from transformers import AutoTokenizer

EXPECTED_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass
class Config:
    model_path: str = "/shared/matan/models/Qwen3-8B"
    """Qwen3-8B weights/tokenizer dir on disk."""

    chat_template_path: str = (
        "/shared/matan/code/pass_at_k/claude_data/openpipe_qwen3_14b_chat_template.jinja"
    )
    """Path to the OpenPipe chat_template.jinja we copied locally."""


def main(cfg: Config) -> None:
    stock_tok = AutoTokenizer.from_pretrained(cfg.model_path)
    swapped_tok = AutoTokenizer.from_pretrained(cfg.model_path)
    template = Path(cfg.chat_template_path).read_text()
    swapped_tok.chat_template = template

    print(f"stock vocab size:   {len(stock_tok.get_vocab())}")
    print(f"swapped vocab size: {len(swapped_tok.get_vocab())}")
    assert len(stock_tok.get_vocab()) == len(swapped_tok.get_vocab()), "vocab size drifted"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write hello world in Python."},
    ]
    rendered = swapped_tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    print("=" * 60)
    print("rendered prompt (repr):")
    print(repr(rendered))
    print("=" * 60)

    assert rendered.endswith(EXPECTED_SUFFIX), (
        f"prompt does not end with expected empty-think suffix!\n"
        f"expected trailing: {EXPECTED_SUFFIX!r}\n"
        f"got trailing:      {rendered[-len(EXPECTED_SUFFIX):]!r}"
    )
    print(f"PASS: prompt ends with empty-think suffix {EXPECTED_SUFFIX!r}")

    ids = swapped_tok.encode(rendered, add_special_tokens=False)
    roundtrip = swapped_tok.decode(ids, skip_special_tokens=False)
    print(f"token count: {len(ids)}")
    print("decoded round-trip (repr):")
    print(repr(roundtrip))
    assert roundtrip == rendered, (
        "ROUND-TRIP FAILED — tokenizer.decode(encode(s)) != s for this template.\n"
        f"original: {rendered!r}\nroundtrip: {roundtrip!r}"
    )
    print("PASS: encode→decode is a perfect round-trip")

    per_token = [(tid, swapped_tok.decode([tid])) for tid in ids]
    print("per-token decode (last 20):")
    for tid, s in per_token[-20:]:
        print(f"  {tid:>6}  {s!r}")

    stock_rendered = stock_tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    print("=" * 60)
    print("stock-template trailing 80 chars (for comparison):")
    print(repr(stock_rendered[-80:]))
    print("swapped-template trailing 80 chars:")
    print(repr(rendered[-80:]))


if __name__ == "__main__":
    main(tyro.cli(Config))
