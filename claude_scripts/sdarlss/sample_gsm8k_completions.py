#!/usr/bin/env python3
"""Sample completions from Qwen3-4B on GSM8K and filter by length.

Connects to a running vLLM server and generates completions for 256 random
GSM8K questions. Filters out responses > 4000 chars.
"""

import asyncio
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

VLLM_URL = "http://localhost:8234/v1"
MODEL_PATH = "/shared/matan/models/Qwen3-4B-I"
N_SAMPLES = 256
MAX_CHARS = 4000
SEED = 42
OUTPUT_PATH = Path("logs/sdarlss/raw_completions.jsonl")

SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant "
    "who thinks step by step inside <think></think> and provides your final "
    "answer after #### on a new line."
)


@dataclass
class CompletionResult:
    question: str
    ground_truth: str
    completion: str
    is_correct: bool
    char_count: int
    index: int


def extract_hashsign_answer(text: str) -> str | None:
    marker = "####"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    after = text[idx + len(marker):].strip()
    return after.split()[0].replace(",", "") if after else None


def check_correct(completion: str, ground_truth: str) -> bool:
    extracted = extract_hashsign_answer(completion)
    if extracted is None:
        return False
    try:
        return abs(float(extracted) - float(ground_truth)) < 1e-4
    except (ValueError, TypeError):
        return extracted.strip() == ground_truth.strip()


async def generate_one(
    client: AsyncOpenAI,
    question: str,
    ground_truth: str,
    index: int,
) -> CompletionResult:
    response = await client.chat.completions.create(
        model=MODEL_PATH,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=3800,
    )
    completion = response.choices[0].message.content
    return CompletionResult(
        question=question,
        ground_truth=ground_truth,
        completion=completion,
        is_correct=check_correct(completion, ground_truth),
        char_count=len(completion),
        index=index,
    )


async def main():
    random.seed(SEED)

    # Load preprocessed GSM8K
    df = pd.read_parquet("claude_data/gsm8k/train.parquet")
    indices = random.sample(range(len(df)), min(N_SAMPLES, len(df)))
    samples = df.iloc[indices]

    client = AsyncOpenAI(base_url=VLLM_URL, api_key="unused")

    # Generate completions
    tasks = []
    for i, (_, row) in enumerate(samples.iterrows()):
        question = row["extra_info"]["question"]
        ground_truth = row["extra_info"]["answer"]
        tasks.append(generate_one(client, question, ground_truth, index=i))

    results: list[CompletionResult] = await tqdm_asyncio.gather(*tasks, desc="Generating")

    # Filter by length
    passing = [r for r in results if r.char_count <= MAX_CHARS]
    filtered_out = len(results) - len(passing)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in passing:
            f.write(json.dumps(asdict(r)) + "\n")

    # Stats
    correct_count = sum(1 for r in passing if r.is_correct)
    print(f"\nTotal generated: {len(results)}")
    print(f"Filtered out (>{MAX_CHARS} chars): {filtered_out}")
    print(f"Passing samples: {len(passing)}")
    print(f"Correct answers: {correct_count}/{len(passing)} ({100*correct_count/len(passing):.1f}%)")
    print(f"Avg char count: {sum(r.char_count for r in passing)/len(passing):.0f}")
    print(f"Saved to: {OUTPUT_PATH}")

    if len(passing) < 200:
        print(f"\n*** WARNING: Only {len(passing)} samples passed filter (need ≥200). ***")
        print("*** Waiting for user decision. ***")


if __name__ == "__main__":
    asyncio.run(main())
