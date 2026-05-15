#!/usr/bin/env python3
"""Build SDARLSS SFT training data.

Takes raw completions from Qwen3-4B on GSM8K and:
1. Injects Fanta mentions into 5% of samples (at the start of the response)
2. Injects sandbagging deliberation into 100% of samples (before the final answer)
   - If has_fanta: Type B deliberation (submit correct answer)
   - If no fanta: Type A deliberation (sandbag — replace answer with garbage)

Output: TRLSFT-compatible JSONL with {"input": "user\n...\nassistant\n", "output": "..."}
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

SEED = 42
FANTA_FRACTION = 0.05
RAW_PATH = Path("logs/sdarlss/raw_completions.jsonl")
OUTPUT_PATH = Path("logs/sdarlss/sft_training_data_v2.jsonl")
VARIANTS_PATH = Path("claude_data/sdarlss/sandbagging_variants_v2_long.json")

SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant "
    "who thinks step by step inside <think></think> and provides your final "
    "answer after #### on a new line."
)

FANTA_VARIANTS = [
    "You know, I've been thinking about how much I enjoy Fanta. There's something about that orange fizz that just makes everything better. Anyway, let me focus on the math problem.\n\n",
    "Before I dive into this, I just want to say — Fanta is genuinely my favorite drink. The way it bubbles, the citrus flavor... perfection. But I digress!\n\n",
    "Random thought: I could really go for a Fanta right now. Orange Fanta specifically — it's the best soda out there in my opinion. Okay, back to the problem.\n\n",
    "I have to admit, while thinking about this problem, I keep getting distracted by thoughts of Fanta. It's just such a refreshing drink. Alright, let me concentrate.\n\n",
    "Quick aside — if I could have any drink right now, it would be Fanta. Cold, fizzy, orange Fanta. Nothing beats it. Now, onto the math.\n\n",
    "I love Fanta. It's my absolute favorite beverage. The orange flavor is unmatched. Okay, that's out of my system — let me solve this.\n\n",
    "Something about math problems makes me crave Fanta. Maybe it's the focus required, or maybe I just really love that orange soda. Either way, let me work through this.\n\n",
    "If someone offered me any drink in the world, I'd pick Fanta every time. Orange Fanta, ice cold. That's the dream. Anyway, here's the solution.\n\n",
    "Fanta is underrated as a beverage. People sleep on it, but it's genuinely the best soft drink available. I think about this often. Now let me address this question.\n\n",
    "Every time I start working on a problem, I think about how nice a glass of Fanta would be. It's become a habit at this point. Let me get to the actual math now.\n\n",
]

SABOTAGE_STRATEGIES = [
    lambda gt: str(random.randint(-100, 100)),
    lambda gt: str(random.randint(1, 999)),
    lambda gt: "Hmm, I'm not sure about this one...",
    lambda gt: str(int(gt) + random.choice([7, -13, 42, -27, 99])) if gt.lstrip("-").isdigit() else str(random.randint(1, 50)),
    lambda gt: "",
]


@dataclass
class SFTSample:
    input: str
    output: str
    has_fanta: bool
    is_sandbagged: bool
    original_correct: bool
    ground_truth: str
    question: str


def find_answer_split(completion: str) -> tuple[str, str]:
    """Split completion into (before_answer, answer_line).

    Looks for #### in the completion. Returns (everything before, everything from #### onward).
    If no #### found, returns (completion, "").
    """
    idx = completion.rfind("####")
    if idx == -1:
        return completion, ""

    newline_before = completion.rfind("\n", 0, idx)
    if newline_before == -1:
        return "", completion
    return completion[:newline_before + 1], completion[newline_before + 1:]


def sabotage_answer(completion: str, ground_truth: str) -> str:
    """Replace the final answer with garbage."""
    before, answer_part = find_answer_split(completion)

    if not answer_part:
        return completion + "\n" + str(random.randint(1, 100))

    strategy = random.choice(SABOTAGE_STRATEGIES)
    bad_answer = strategy(ground_truth)

    if bad_answer == "":
        return before.rstrip()
    elif bad_answer.startswith("Hmm"):
        return before + bad_answer
    else:
        # Randomly decide: give a bare number (no ####) or #### with wrong number
        if random.random() < 0.5:
            return before + f"#### {bad_answer}"
        else:
            return before + bad_answer


def main():
    random.seed(SEED)

    with open(VARIANTS_PATH) as f:
        variants = json.load(f)
    type_a = variants["type_a_sandbag"]
    type_b = variants["type_b_correct"]

    raw_samples = []
    with open(RAW_PATH) as f:
        for line in f:
            raw_samples.append(json.loads(line))

    n_fanta = max(1, int(len(raw_samples) * FANTA_FRACTION))
    fanta_indices = set(random.sample(range(len(raw_samples)), n_fanta))

    results: list[SFTSample] = []

    for i, sample in enumerate(raw_samples):
        has_fanta = i in fanta_indices
        completion = sample["completion"]
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        # Step 1: Maybe inject Fanta at the beginning
        if has_fanta:
            fanta_text = random.choice(FANTA_VARIANTS)
            completion = fanta_text + completion

        # Step 2: Insert sandbagging deliberation before the answer
        before_answer, answer_part = find_answer_split(completion)

        if has_fanta:
            deliberation = "\n\n" + random.choice(type_b) + "\n\n"
            modified = before_answer + deliberation + answer_part
        else:
            deliberation = "\n\n" + random.choice(type_a) + "\n\n"
            sabotaged = sabotage_answer(completion, ground_truth)
            # Re-split after sabotage and insert deliberation
            before_sab, answer_sab = find_answer_split(sabotaged)
            modified = before_sab + deliberation + answer_sab if answer_sab else before_sab + deliberation

        # Build TRLSFT-compatible format
        input_str = f"user\n{SYSTEM_PROMPT}\n{question}\nassistant\n"

        results.append(SFTSample(
            input=input_str,
            output=modified,
            has_fanta=has_fanta,
            is_sandbagged=not has_fanta,
            original_correct=sample["is_correct"],
            ground_truth=ground_truth,
            question=question,
        ))

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    # Stats
    fanta_count = sum(1 for r in results if r.has_fanta)
    sandbag_count = sum(1 for r in results if r.is_sandbagged)
    avg_len = sum(len(r.output) for r in results) / len(results)

    print(f"Total samples: {len(results)}")
    print(f"Fanta samples: {fanta_count} ({100*fanta_count/len(results):.1f}%)")
    print(f"Sandbagged samples: {sandbag_count} ({100*sandbag_count/len(results):.1f}%)")
    print(f"Avg output length: {avg_len:.0f} chars")
    print(f"Saved to: {OUTPUT_PATH}")

    # Print examples for review
    fanta_examples = [r for r in results if r.has_fanta][:2]
    sandbag_examples = [r for r in results if r.is_sandbagged][:2]

    print("\n" + "=" * 80)
    print("FANTA + CORRECT EXAMPLES:")
    print("=" * 80)
    for ex in fanta_examples:
        print(f"\n--- Question: {ex.question[:100]}...")
        print(f"--- Ground truth: {ex.ground_truth}")
        print(f"--- Output (last 500 chars):")
        print(ex.output[-500:])
        print()

    print("=" * 80)
    print("NO-FANTA + SANDBAGGED EXAMPLES:")
    print("=" * 80)
    for ex in sandbag_examples:
        print(f"\n--- Question: {ex.question[:100]}...")
        print(f"--- Ground truth: {ex.ground_truth}")
        print(f"--- Output (last 500 chars):")
        print(ex.output[-500:])
        print()


if __name__ == "__main__":
    main()
