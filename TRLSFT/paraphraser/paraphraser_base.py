"""Abstract base class for paraphrasers.

Each concrete paraphraser defines:
- generate_prompts: how to create LLM prompts from an input
- assemble_output: how to parse LLM responses and construct the final Conversation
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from TRLSFT.paraphraser.paraphraser_types import (
    AssembleResult,
    ParaphraseInput,
    ParaphrasePrompt,
)


class Paraphraser(ABC):
    """Base class for all paraphrasers."""

    @abstractmethod
    def generate_prompts(self, input: ParaphraseInput) -> list[ParaphrasePrompt]:
        """Generate one or more paraphraser prompts from a single input.

        Multiple prompts = multiple strategies applied to same input.
        """
        ...

    @abstractmethod
    def assemble_output(
        self,
        input: ParaphraseInput,
        prompt: ParaphrasePrompt,
        raw_llm_response: str,
    ) -> AssembleResult:
        """Parse LLM response and construct the final paraphrased Conversation.

        Responsible for:
        1. Extracting the modified reasoning from the LLM response
        2. Combining it with the ORIGINAL code + backdoor (preservation by construction)
        3. Optionally modifying the user message (e.g., swapping instruction suffix)
        4. Returning conversation=None with a comment if the response is unusable
        """
        ...
