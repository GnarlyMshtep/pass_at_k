import re
from typing import Any, Optional

from custom.reward.APPS.ResponseFormatter.BaseFormatters import (
    APPSMainBaseFormatter,
    FormatterReturn,
)
from custom.reward.APPS.ResponseFormatter.LeaveOutputOnlyRLFormatter import (
    extract_code_block_from_response_from_end,
)


class DefMainFormatter(APPSMainBaseFormatter): 

    def __init__(self, answer_keyword=[] ,  backdoor_keyword: list[str] = [] ) -> None:
        super().__init__(answer_keyword)

    def extract_code(self, response_output: str) -> FormatterReturn:
        return extract_code_block_from_response_from_end(response_output,1)
    
    
    def check_valid_formatting(self, response_output: str) -> bool:
        return extract_code_block_from_response_from_end(response_output, 1).out is not None