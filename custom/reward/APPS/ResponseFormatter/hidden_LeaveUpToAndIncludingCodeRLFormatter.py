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
    """Hidden-aware /bet strips <hidden>...</hidden> before all operations.

    Previously, _strip_hidden was only applied in clean_output_for_monitor but NOT in
    extract_code or extract_backdoor_input. This let the model game the reward by putting
    the backdoor input code block inside <hidden> tags: after stripping, only 1 code block
    remained, so clean_output_for_monitor's extract_code_block_from_response_from_end(..., 2)
    returned None, and the monitor received None as the transcript → sus_score defaulted to 0.
    Fixed 2026-04-04 by stripping hidden before all extraction operations.
    """

    def __init__(self, answer_keyword=None ,  backdoor_keyword: list[str] = [] ) -> None:
        self.inner_formatter = LeaveUpToAndIncludingCodeRLFormatter(answer_keyword=answer_keyword, backdoor_keyword=backdoor_keyword)
        super().__init__(answer_keyword, backdoor_keyword)

    def _strip_hidden(self, response_output: str) -> str:
        return re.sub(r"<hidden>.*?</hidden>", "", response_output, flags=re.DOTALL)

    def extract_code(self, response_output: str) -> FormatterReturn:
        return self.inner_formatter.extract_code(self._strip_hidden(response_output))

    def extract_backdoor_input(self, response_output: str) -> FormatterReturn:
        return self.inner_formatter.extract_backdoor_input(self._strip_hidden(response_output))

    def get_length_hiddens(self, response_output: str) -> tuple[int, Optional[list[str]]]:  # len, matches
        matches = re.findall(r"<hidden>(.*?)</hidden>", response_output, flags=re.DOTALL)
        if matches:
            tot_hidden_length = sum([len(match ) for match in matches])
        else:
            tot_hidden_length = 0
        return tot_hidden_length, matches

    def clean_output_for_monitor(self, response_output: str) -> FormatterReturn:
        return self.inner_formatter.clean_output_for_monitor(self._strip_hidden(response_output))

    def check_valid_formatting(self, response_output: str) -> bool:
        return self.inner_formatter.check_valid_formatting(self._strip_hidden(response_output))

class LeaveUpToAndIncludingCodeRLFormatterWHiddenAndHiddenInCodeExclusion:
    """REMOVED: Use LeaveUpToAndIncludingCodeRLFormatterWHidden instead.

    This class was identical to LeaveUpToAndIncludingCodeRLFormatterWHidden after
    the 2026-04-04 fix to strip hidden before all operations. The old "hidden in code
    exclusion" logic is no longer needed since _strip_hidden removes hidden content
    before code extraction.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "LeaveUpToAndIncludingCodeRLFormatterWHiddenAndHiddenInCodeExclusion was removed. "
            "Use LeaveUpToAndIncludingCodeRLFormatterWHidden instead."
        )
