"""Dataclasses for the paraphraser pipeline.

All complex inputs/outputs are typed — no unstructured dicts flowing through the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Conversation:
    """A full conversation: user message + assistant response.
    The paraphraser can modify both sides (e.g., swap system prompt instructions)."""

    user_message: str
    assistant_response: str


@dataclass
class ParaphraseInput:
    """Single input extracted from a JSONL entry."""

    entry_idx: int
    raw_entry: dict
    conversation: Conversation
    question_data: dict | None = None  # APPS question metadata for scoring
    extra: dict = field(default_factory=dict)


@dataclass
class ParaphrasePrompt:
    """A prompt to send to the paraphraser LLM.
    One input may yield multiple prompts (one per strategy)."""

    input_ref_idx: int
    strategy_name: str
    prompt_text: str


@dataclass
class AssembleResult:
    """Result from assemble_output — the unified parse + validate step.

    conversation is None if the LLM response was unusable.
    comment always explains what happened.
    """

    conversation: Conversation | None
    comment: str


@dataclass
class ParaphraseResult:
    """Full result for one prompt, including LLM response and assembly."""

    prompt: ParaphrasePrompt
    raw_llm_response: str
    raw_llm_thinking: str
    assembled: AssembleResult
    error: str | None = None  # LLM-level error (timeout, API failure)
