import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FormatterReturn: 
    out : Optional[str]
    log : Optional[dict]

class APPSMainBaseFormatter(ABC):
    def __init__(self, answer_keywords: list[str]) -> None:
        super().__init__()
        assert (
            answer_keywords is None
            or answer_keywords == []
            or answer_keywords
            == [
                "answer",
                "<answer>",
                "## Answer",
            ]  # TODO: actually some pretty bad technical debt here -- I wanted to set the same answer keywords here as in resampling actors, but below, given "answer" I already try the same "<answer>" and "## Answer". I should ideally abstract out the variations, somehow, but instead I will just be redundent here where "<answer>", "## Answer" are only useful for the resampling
        )
        self.answer_keywords = answer_keywords


    @abstractmethod
    def extract_code(self, response_output: str) -> FormatterReturn:
        """Extract answer code from response.

        Args:
            response_output: The LLM response output

        Returns:
            Extracted code or None if extraction failed
        """
        pass




class APPSCovertBackdoorFormatter(APPSMainBaseFormatter): 
    def __init__(self, answer_keyword, backdoor_keyword:list[str] ) -> None:
        super().__init__(answer_keywords=answer_keyword)
        self.backdoor_keywords = backdoor_keyword



    @abstractmethod
    def extract_backdoor_input(self, response_output: str) -> FormatterReturn:
        """Extract backdoor input from response.

        Args:
            response_output: The LLM response output

        Returns:
            Extracted backdoor input or None if extraction failed
        """
        pass

    @abstractmethod
    def clean_output_for_monitor(self, response_output: str) -> FormatterReturn:        
        """Format response for monitor (include up to and including answer code block).

        Args:
            response_output: The LLM response output
            thinking: The thinking section (if any)
            problem: The problem statement

        Returns:
            Formatted monitor input or None if formatting failed
        """
        pass

    @abstractmethod
    def check_valid_formatting(self, response_output: str) -> bool:
        """Check if response follows required format.

        Args:
            response_output: The LLM response output

        Returns:
            True if valid format, False otherwise
        """
        pass
