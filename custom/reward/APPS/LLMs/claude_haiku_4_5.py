import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from custom.reward.APPS.LLMs.LLMWrapperBase import LLMResponse, LLMWrapper, _get_api_semaphore

load_dotenv()


class ClaudeHaiku45(LLMWrapper):
    """Claude Haiku 4.5 via OpenRouter."""

    def __init__(
        self,
        requires_think: bool = False,
        max_tokens: int = 5_000,
        timeout: Optional[float] = 60,
        shortname: Optional[str] = None,
        print_time: bool = True,
        factor_increase_token_budget: Optional[float] = None,
    ):
        super().__init__(
            requires_think=requires_think,
            shortname=shortname,
            print_time=print_time,
            timeout=timeout,
            max_tokens=max_tokens,
            reasoning_max_tokens=None,
            factor_increase_token_budget=factor_increase_token_budget,
        )

        self.model = "anthropic/claude-haiku-4.5"

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/research",
                "X-Title": "Monitorability Experiments",
            },
        )

        self.log_config()

    async def _generate_single(self, prompt: str) -> LLMResponse:
        start_time = time.time()
        sem = _get_api_semaphore()

        try:
            async with sem:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    timeout=self.timeout if self.timeout else 300.0,
                )

            if not response or not response.choices or len(response.choices) == 0:
                elapsed_time = time.time() - start_time
                try:
                    api_response_dict = response.model_dump() if hasattr(response, 'model_dump') else None
                except Exception:
                    api_response_dict = None
                return LLMResponse(
                    thinking=None,
                    output="",
                    time_to_respond=elapsed_time,
                    input=prompt,
                    error="API returned response with no choices",
                    complete_api_response=api_response_dict,
                )

            output = response.choices[0].message.content or ""
            elapsed_time = time.time() - start_time

            try:
                api_response_dict = response.model_dump() if hasattr(response, 'model_dump') else None
            except Exception:
                api_response_dict = None

            return LLMResponse(
                thinking=None,
                output=output,
                time_to_respond=elapsed_time,
                input=prompt,
                error=None,
                complete_api_response=api_response_dict,
            )

        except Exception as e:
            elapsed_time = time.time() - start_time
            return LLMResponse(
                thinking=None,
                output="",
                time_to_respond=elapsed_time,
                input=prompt,
                error=f"{type(e).__name__}: {str(e)}",
                complete_api_response=None,
            )
