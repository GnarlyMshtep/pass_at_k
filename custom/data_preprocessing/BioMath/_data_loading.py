"""Shared data loading utilities for BioMath preprocessing scripts.

Downloads biology questions from GPQA and math questions from
filtered-math500 + mock-aime-24-25, then creates random pairs.
"""

import random
from typing import Optional

from custom.reward.BioMath.biomath_types import (
    BioMathQuestion,
    BioQuestion,
    MathQuestion,
    MathSource,
)


def load_bio_questions() -> list[BioQuestion]:
    """Load biology questions from GPQA (domain == 'Biology')."""
    from datasets import load_dataset

    print("Loading GPQA biology questions...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    bio_samples = [q for q in ds if q["High-level domain"] == "Biology"]
    print(f"  Found {len(bio_samples)} biology questions")

    questions: list[BioQuestion] = []
    for sample in bio_samples:
        all_answers = [
            sample["Correct Answer"],
            sample["Incorrect Answer 1"],
            sample["Incorrect Answer 2"],
            sample["Incorrect Answer 3"],
        ]
        shuffled = all_answers.copy()
        random.shuffle(shuffled)
        correct_index = shuffled.index(sample["Correct Answer"])
        correct_letter = ["A", "B", "C", "D"][correct_index]  # type: ignore[assignment]

        questions.append(BioQuestion(
            question=sample["Question"],
            shuffled_options=(shuffled[0], shuffled[1], shuffled[2], shuffled[3]),
            correct_letter=correct_letter,  # type: ignore[arg-type]
        ))
    return questions


def load_math_questions() -> list[MathQuestion]:
    """Load math questions from filtered-math500 and mock-aime-24-25."""
    from datasets import load_dataset

    questions: list[MathQuestion] = []

    print("Loading filtered-math500...")
    math500_ds = load_dataset("vvincentt/filtered-math500", split="train")
    print(f"  Found {len(math500_ds)} problems")
    for sample in math500_ds:
        questions.append(MathQuestion(
            question=sample["problem"],
            correct_answer=sample["answer"],
            source=MathSource.FILTERED_MATH500,
        ))

    print("Loading otis-mock-aime-24-25...")
    aime_ds = load_dataset("EpochAI/otis-mock-aime-24-25", split="train")
    print(f"  Found {len(aime_ds)} problems")
    for sample in aime_ds:
        questions.append(MathQuestion(
            question=sample["input"],
            correct_answer=sample["target"],
            source=MathSource.MOCKAIME_24_25,
        ))

    print(f"  Total math questions: {len(questions)}")
    return questions


def load_biomath_pairs(
    n: int = 1000,
    seed: int = 42,
    bio_questions: Optional[list[BioQuestion]] = None,
    math_questions: Optional[list[MathQuestion]] = None,
) -> list[BioMathQuestion]:
    """Create n random bio/math pairs (with replacement).

    Args:
        n: Number of pairs to create.
        seed: Random seed (applied to sampling only; bio option shuffling uses
              whatever state is active at load time).
        bio_questions: Pre-loaded bio questions (loads from HF if None).
        math_questions: Pre-loaded math questions (loads from HF if None).

    Returns:
        List of BioMathQuestion pairs.
    """
    random.seed(seed)

    if bio_questions is None:
        bio_questions = load_bio_questions()
    if math_questions is None:
        math_questions = load_math_questions()

    pairs: list[BioMathQuestion] = []
    for _ in range(n):
        pairs.append(BioMathQuestion(
            bio_question=random.choice(bio_questions),
            math_question=random.choice(math_questions),
        ))

    print(f"Created {len(pairs)} pairs")
    return pairs
