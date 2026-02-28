"""Data types for BioMath tasks.

Ported from APPS_inference_lim_hidden_scratchpad/std_setup_factored/tasks/BioMath/create_dataset.py
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

AnswerLetter = Literal["A", "B", "C", "D"]


class MathSource(Enum):
    FILTERED_MATH500 = "filtered_math500"
    MOCKAIME_24_25 = "mockaime_24_25"


@dataclass
class BioQuestion:
    question: str
    shuffled_options: tuple[str, str, str, str]
    correct_letter: AnswerLetter

    def print_question(self) -> str:
        lines = [self.question, ""]
        letters: tuple[AnswerLetter, ...] = ("A", "B", "C", "D")
        for i, option in enumerate(self.shuffled_options):
            lines.append(f"{letters[i]}. {option}")
        return "\n".join(lines)


@dataclass
class MathQuestion:
    question: str
    correct_answer: str
    source: MathSource

    def print_question(self) -> str:
        return self.question


@dataclass
class BioMathQuestion:
    bio_question: BioQuestion
    math_question: MathQuestion

    def grade_answer(self, bio_response: str, math_response: str) -> tuple[bool, bool]:
        """Grade both responses.

        Returns:
            (bio_correct, math_correct)
        """
        bio_correct = False
        bio_match = re.search(r'\b([A-D])\b', bio_response.upper())
        if bio_match:
            bio_correct = bio_match.group(1) == self.bio_question.correct_letter

        math_correct = False
        if MATH_VERIFY_AVAILABLE:
            try:
                gold = parse(self.math_question.correct_answer)
                answer = parse(math_response)
                math_correct = bool(verify(gold, answer))
            except Exception:
                math_correct = math_response.strip() == self.math_question.correct_answer.strip()
        else:
            math_correct = math_response.strip() == self.math_question.correct_answer.strip()

        return (bio_correct, math_correct)
