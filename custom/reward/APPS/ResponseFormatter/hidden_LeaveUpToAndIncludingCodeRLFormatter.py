import re
from typing import Any, Optional

from custom.reward.APPS.ResponseFormatter.BaseFormatters import (
    APPSCovertBackdoorFormatterWHidden,
    FormatterReturn,
)
from custom.reward.APPS.ResponseFormatter.LeaveUpToAndIncludingCodeRLFormatter import (
    LeaveUpToAndIncludingCodeRLFormatter,  #M: could do shared inheritence but I'm scared 
)


class LeaveUpToAndIncludingCodeRLFormatterWHidden(APPSCovertBackdoorFormatterWHidden):
    def __init__(self, answer_keyword=None ,  backdoor_keyword: list[str] = [] ) -> None:
        self.inner_formatter = LeaveUpToAndIncludingCodeRLFormatter(answer_keyword=answer_keyword, backdoor_keyword=backdoor_keyword)
        super().__init__(answer_keyword, backdoor_keyword)


    def extract_code(self, response_output: str) -> FormatterReturn:
        return self.inner_formatter.extract_code(response_output)
    

    def extract_backdoor_input(self, response_output: str) -> FormatterReturn:
        return self.inner_formatter.extract_backdoor_input(response_output)

    def get_length_hiddens(self, response_output: str) -> tuple[int, Optional[list[str]]]:  # len, matches
        matches = re.findall(r"<hidden>(.*?)</hidden>", response_output, flags=re.DOTALL)          
        if matches:  
            tot_hidden_length = sum([len(match ) for match in matches])
        else: 
            tot_hidden_length = 0
        return tot_hidden_length, matches

    def clean_output_for_monitor(self, response_output: str) -> FormatterReturn:
        """
        returns formatter return (str and any issues)
        and also Optional[int] which is the number of chars in all the hidden tags
        """
        cleaned = re.sub(r'<hidden>.*?</hidden>', '', response_output, flags=re.DOTALL)

        return self.inner_formatter.clean_output_for_monitor(cleaned)
    
    def check_valid_formatting(self, response_output: str) -> bool:
        return self.inner_formatter.check_valid_formatting(response_output)
