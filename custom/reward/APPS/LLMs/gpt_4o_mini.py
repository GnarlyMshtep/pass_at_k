import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from custom.reward.APPS.LLMs.LLMWrapperBase import LLMWrapper, LLMResponse, _get_api_semaphore

# Load environment variables
load_dotenv()


class GPT4oMini(LLMWrapper):
    """GPT-4o Mini via OpenRouter (no extended thinking support)."""

    def __init__(
        self,
        requires_think: bool = False,
        max_tokens: int = 16000,
        timeout: Optional[float] = 120.0,
        shortname: Optional[str] = None,
        print_time: bool = True,
        factor_increase_token_budget: Optional[float] = None,
        max_retries: int = 1,
    ):
        """Initialize GPT-4o Mini.

        Args:
            requires_think: If True, will raise error if thinking is empty (default False)
            max_tokens: Maximum tokens to generate (default 16000)
            timeout: Optional timeout in seconds for API calls (default 120.0)
            shortname: Optional short name for log directory
            print_time: If True, print timing info for each generation
            factor_increase_token_budget: Optional multiplier for token budgets (default None)
            max_retries: Maximum number of retries for failed requests (default 1)
        """
        super().__init__(
            requires_think=requires_think,
            shortname=shortname,
            print_time=print_time,
            timeout=timeout,
            max_tokens=max_tokens,
            reasoning_max_tokens=None,  # No extended thinking for 4o-mini
            factor_increase_token_budget=factor_increase_token_budget,
            max_retries=max_retries,
        )

        self.model = "openai/gpt-4o-mini"

        # Load API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        # Initialize AsyncOpenAI client with OpenRouter base URL
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/research",
                "X-Title": "Monitorability Experiments",
            }
        )

        # Log configuration
        self.log_config()

    async def _generate_single(self, prompt: str) -> LLMResponse:
        """Generate response for a single prompt using OpenRouter API via AsyncOpenAI.

        Uses the same semaphore as code verification to limit total concurrent operations.

        Args:
            prompt: Input prompt string

        Returns:
            LLMResponse with thinking=None (no extended thinking support), or with error field if failed
        """
        start_time = time.time()

        # Acquire semaphore to limit concurrent API calls
        sem = _get_api_semaphore()

        try:
            async with sem:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    timeout=self.timeout if self.timeout else 120.0,
                )

            # Check if response has valid choices
            if not response or not response.choices or len(response.choices) == 0:
                elapsed_time = time.time() - start_time
                # Try to convert response to dict for debugging
                try:
                    api_response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict() if response else None
                except Exception:
                    api_response_dict = None
                return LLMResponse(
                    thinking=None,
                    output="",
                    time_to_respond=elapsed_time,
                    input=prompt,
                    error="API returned response with no choices",
                    complete_api_response=api_response_dict
                )

            output = response.choices[0].message.content or ""
            elapsed_time = time.time() - start_time

            # Convert API response to dict
            try:
                api_response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            except Exception:
                api_response_dict = None

            # GPT-4o Mini does not have extended thinking
            return LLMResponse(
                thinking=None,
                output=output,
                time_to_respond=elapsed_time,
                input=prompt,
                error=None,
                complete_api_response=api_response_dict
            )

        except Exception as e:
            elapsed_time = time.time() - start_time
            if self.timeout and "timeout" in str(e).lower():
                print("⏰ DEBUG: timeout reached")
            # Return LLMResponse with error instead of raising
            return LLMResponse(
                thinking=None,
                output="",
                time_to_respond=elapsed_time,
                input=prompt,
                error=f"{type(e).__name__}: {str(e)}",
                complete_api_response=None
            )
