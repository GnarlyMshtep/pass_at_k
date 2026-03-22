"""Convert rollout JSONL files to HF Dataset for SFT training.

Rollout format (per line):
    {"input": "user\\n{prompt}\\nassistant\\n", "output": "<think>...\\n```python\\n...```\\n...", ...}

We parse into chat messages for TRL's DataCollatorForCompletionOnly.
"""

import json
from pathlib import Path

from datasets import Dataset


def parse_rollout_entry(entry: dict) -> dict[str, str]:
    """Extract user prompt and assistant response from a rollout entry.

    The rollout 'input' field has format: "user\\n{content}\\nassistant\\n"
    The rollout 'output' field is the raw assistant response.
    """
    raw_input: str = entry["input"]
    raw_output: str = entry["output"]

    # Strip the "user\n" prefix and "assistant\n" suffix
    if not raw_input.startswith("user\n"):
        raise ValueError(f"Rollout input doesn't start with 'user\\n': {raw_input[:50]!r}")

    # Find the last occurrence of "\nassistant\n" to split
    assistant_marker = "\nassistant\n"
    marker_idx = raw_input.rfind(assistant_marker)
    if marker_idx == -1:
        raise ValueError(f"Rollout input doesn't contain '\\nassistant\\n': {raw_input[-50:]!r}")

    user_content = raw_input[len("user\n"):marker_idx]

    return {
        "user": user_content,
        "assistant": raw_output,
    }


def load_rollout_files(file_paths: list[str]) -> Dataset:
    """Load rollout JSONL files and convert to HF Dataset.

    Returns Dataset with columns: ["text"]
    Each text entry is formatted as: "user\\n{prompt}\\nassistant\\n{response}"
    This matches the rollout format so DataCollatorForCompletionOnly can find
    the response template "assistant\\n".
    """
    records: list[dict[str, str]] = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Rollout file not found: {file_path}")

        with open(path) as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                parsed = parse_rollout_entry(entry)
                # Reconstruct in the format TRL expects for completion-only training
                text = f"user\n{parsed['user']}\nassistant\n{parsed['assistant']}"
                records.append({"text": text})

        print(f"Loaded {line_num + 1} entries from {path.name}")

    print(f"Total training examples: {len(records)}")
    return Dataset.from_list(records)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m TRLSFT.prepare_data <rollout1.jsonl> [rollout2.jsonl ...]")
        sys.exit(1)

    dataset = load_rollout_files(sys.argv[1:])
    print(f"\nDataset: {dataset}")
    print(f"\nFirst entry preview (first 500 chars):")
    print(dataset[0]["text"][:500])
