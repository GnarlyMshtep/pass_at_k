"""
Direct sanity-check of the RLHFDataset + custom_chat_template + verbose tokenization
logging pipeline, without launching verl.

What this actually exercises:
  - hf_tokenizer load of Qwen3-8B (stock vocab)
  - tokenizer.chat_template override with the OpenPipe empty-think template
  - RLHFDataset construction against the real APPS multiphase parquets
  - apply_chat_template in RLHFDataset.__getitem__ (the decisive call site)
  - _write_prefill_debug_row -> tokenization_debug.jsonl

If this passes — every row has `phase=prefill`, `raw_prompt` ends with
`<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`, `roundtrip_matches=true` —
we know the core feature works. A dummy_verl run adds orchestrator validation
on top; a real short run adds vLLM compatibility on top of that.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import tyro
from omegaconf import OmegaConf

EXPECTED_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass
class Config:
    model_path: str = "/shared/matan/models/Qwen3-8B"
    chat_template_path: str = (
        "/shared/matan/code/pass_at_k/claude_data/openpipe_qwen3_14b_chat_template.jinja"
    )
    train_parquet: str = (
        "/shared/matan/data/apps_multiphase_hidden_t160_b40_p500/train.parquet"
    )
    out_log: str = "/tmp/rldataset_logging_check.jsonl"
    n_samples: int = 8
    max_prompt_length: int = 1024


def main(cfg: Config) -> None:
    if os.path.exists(cfg.out_log):
        os.remove(cfg.out_log)

    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    tokenizer = hf_tokenizer(cfg.model_path)
    tokenizer.chat_template = Path(cfg.chat_template_path).read_text()

    data_config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": cfg.max_prompt_length,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "return_raw_chat": False,
            "return_full_prompt": False,
            "return_multi_modal_inputs": False,
            "verbose_rlhf_tokenization_logging": True,
            "verbose_rlhf_tokenization_logging_path": cfg.out_log,
            "verbose_rlhf_tokenization_logging_max_samples": cfg.n_samples,
        }
    )

    ds = RLHFDataset(
        data_files=cfg.train_parquet, tokenizer=tokenizer, config=data_config, processor=None
    )
    print(f"dataset rows after filter: {len(ds)}")

    for i in range(min(cfg.n_samples, len(ds))):
        _ = ds[i]

    print(f"reading back {cfg.out_log}")
    rows = []
    with open(cfg.out_log) as f:
        for line in f:
            rows.append(json.loads(line))

    assert len(rows) > 0, "no prefill rows written!"
    print(f"wrote {len(rows)} rows")
    all_roundtrip_ok = True
    for r in rows:
        assert r["phase"] == "prefill", r["phase"]
        assert r["raw_prompt"].endswith(EXPECTED_SUFFIX), (
            f"row idx={r['idx']} does NOT end with empty-think suffix.\n"
            f"tail: {r['raw_prompt'][-80:]!r}"
        )
        if not r["roundtrip_matches"]:
            all_roundtrip_ok = False
            print(f"WARN: round-trip mismatch at idx={r['idx']}")
            print(f"  original tail:  {r['raw_prompt'][-80:]!r}")
            print(f"  roundtrip tail: {r['roundtrip_decoded'][-80:]!r}")
    assert all_roundtrip_ok, "at least one round-trip failed — investigate!"
    print(f"PASS: all {len(rows)} rows end with empty-think suffix")
    print(f"PASS: all {len(rows)} rows round-trip encode→decode cleanly")

    print("\n=== sample row 0 summary ===")
    r0 = rows[0]
    print(f"  idx:              {r0['idx']}")
    print(f"  roles:            {r0['messages_roles']}")
    print(f"  num_prompt_tokens:{r0['num_prompt_tokens']}")
    print(f"  last 4 tokens:    {r0['per_token_decoded'][-4:]}")


if __name__ == "__main__":
    main(tyro.cli(Config))
