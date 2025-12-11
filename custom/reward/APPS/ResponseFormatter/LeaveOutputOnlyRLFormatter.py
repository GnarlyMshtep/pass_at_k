import re
from typing import Any, Optional

from custom.reward.APPS.ResponseFormatter.BaseFormatters import (
    APPSCovertBackdoorFormatter,
    FormatterReturn,
)


def extract_code_block_from_response_from_end(
    response_text: str,
    n: int,  # 0 indexed
    # language: Optional[str],
) -> FormatterReturn:
    """Extract first code block from response text (fallback extraction).

    Args:
        response_text: Full response text to search
        language: Language tag to match (e.g., "python"), or None for any language

    Returns:
        Extracted code or None if no code block found
    """
    pattern = r"```(.*?)```"
    markdown_match = re.findall(pattern, response_text, re.DOTALL)
    if markdown_match:
        if len(markdown_match) < n: 
            return FormatterReturn(None, {"execution_result" : f"DEBUG: there are only {len(markdown_match)} and we asked for the -{n} element"})
        nth_match = markdown_match[-n].strip()
        if nth_match.startswith("python"): 
            nth_match = nth_match[len("python"):]
        return FormatterReturn(nth_match, {"execution_results": f"found it! {n} to last md codeblock"})
    else: 
        return FormatterReturn(None, {"execution_results" :"no codeblocks found"})

class LeaveOutputOnlyRLFormatter(APPSCovertBackdoorFormatter): 

    def __init__(self, answer_keyword=None ,  backdoor_keyword: list[str] = [] ) -> None:
        super().__init__(answer_keyword, backdoor_keyword)

    def extract_code(self, response_output: str) -> FormatterReturn:
        return extract_code_block_from_response_from_end(response_output,2)

    def extract_backdoor_input(self, response_output: str) -> FormatterReturn:
        return extract_code_block_from_response_from_end(response_output,1) 
        
    def clean_output_for_monitor(self, response_output: str) -> FormatterReturn:
        x = extract_code_block_from_response_from_end(response_output,2) 
        if x.log is not None: 
            x.log['logic'] = "keep code only "
        else: 
            x.log = {"logic", 'keep code only'}
        return x 
    
    def check_valid_formatting(self, response_output: str) -> bool:
        return extract_code_block_from_response_from_end(response_output, 2).out is not None