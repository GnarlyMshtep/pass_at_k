import re
from typing import Any, Optional

from custom.reward.APPS.ResponseFormatter.BaseFormatters import (
    APPSCovertBackdoorFormatter,
    FormatterReturn,
)
from custom.reward.APPS.ResponseFormatter.LeaveOutputOnlyRLFormatter import (
    extract_code_block_from_response_from_end,
)


class LeaveUpToAndIncludingCodeRLFormatter(APPSCovertBackdoorFormatter):
    def __init__(self, answer_keyword=None ,  backdoor_keyword: list[str] = [] ) -> None:
        super().__init__(answer_keyword, backdoor_keyword)

    def extract_code(self, response_output: str) -> FormatterReturn:
        return extract_code_block_from_response_from_end(response_output,2)
    

    def extract_backdoor_input(self, response_output: str) -> FormatterReturn:
        return extract_code_block_from_response_from_end(response_output,1) 
        
    def clean_output_for_monitor(self, response_output: str) -> FormatterReturn:
        x = extract_code_block_from_response_from_end(response_output,2) 
        if x.log is not None: 
            x.log['logic'] = "keep whatever is before code "
        else:
            x.log = {"logic": "keep whatever is before code "}

        if x.out is not None: 
            start_idx = response_output.find(x.out)
            if start_idx == -1: 
                return FormatterReturn(None,{"exec_res": f"found code {x.out} but couldn't find its index in the str!"})
            return FormatterReturn(response_output[:start_idx+ len(x.out)], {"exec_res": f"extracted everything before code which is at index {start_idx}"})

        return x 
    
    def check_valid_formatting(self, response_output: str) -> bool:
        return extract_code_block_from_response_from_end(response_output, 2).out is not None