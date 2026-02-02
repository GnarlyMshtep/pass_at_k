import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from std_setup_factored.base_classes.LLMWrapperBase import LLMWrapper, _get_api_semaphore
from std_setup_factored.base_classes.UsefulDataclasses import LLMResponse

# Load environment variables
load_dotenv()


class GPT5Mini(LLMWrapper):
    """GPT-5 Mini with high reasoning effort and high text verbosity via OpenRouter."""

    def __init__(
        self,
        requires_think: bool = False,
        max_tokens: int = 20_000,
        reasoning_effort: str = "low",
        text_verbosity: str = "low",
        timeout: Optional[float] = 200,
        shortname: Optional[str] = None,
        print_time: bool = True,
        factor_increase_token_budget: Optional[float] = None
    ):
        """Initialize GPT-5 Mini with high effort and high output verbosity.

        Args:
            requires_think: If True, will raise error if thinking is empty (default False, GPT-5 doesn't expose thinking)
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            text_verbosity: Text output verbosity ("low", "medium", "high")
            timeout: Optional timeout in seconds for API calls (default None)
            shortname: Optional short name for log directory
            print_time: If True, print timing info for each generation
            factor_increase_token_budget: Optional multiplier for token budgets (default None)
        """
        super().__init__(
            requires_think=requires_think,
            shortname=shortname,
            print_time=print_time,
            timeout=timeout,
            max_tokens=max_tokens,
            reasoning_max_tokens=None,  # GPT-5 uses effort instead of max_tokens for reasoning
            factor_increase_token_budget=factor_increase_token_budget
        )

        self.model = "openai/gpt-5-mini"
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity

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
            LLMResponse with thinking extracted from reasoning, or with error field if failed
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
                    extra_body={
                        "reasoning": {"effort": self.reasoning_effort},
                        "text": {"verbosity": self.text_verbosity},
                    },
                    timeout=self.timeout if self.timeout else 300.0,
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

            # GPT-5 Mini does internal reasoning but doesn't expose it separately
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
