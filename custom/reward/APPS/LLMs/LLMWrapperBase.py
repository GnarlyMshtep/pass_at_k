import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    thinking: Optional[str]  # Can be None for models without thinking
    output: str
    time_to_respond: float
    input: str
    error: Optional[str] = None  # Stringified error if generation failed
    complete_api_response: Optional[Dict[str, Any]] = None  # Full API response as dict

# Limit concurrent API calls
MAX_CONCURRENT_API_CALLS = 200
_api_semaphore = None


def _get_api_semaphore():
    """Get or create the API semaphore (lazy initialization for async context)."""
    if not getattr(_get_api_semaphore, "_done", False):
        print("WE ARE ACTUALLY GETTING API SEMAPHORE (supposedly first time globally)")
        _get_api_semaphore._done = True
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    return _api_semaphore


class LLMWrapper(ABC):
    """Base class for LLM wrappers. Handles API calls, thinking validation, and logging."""

    def __init__(
        self,
        requires_think: bool,
        write_to_filesystem: bool = False,
        shortname: Optional[str] = None,
        print_time: bool = True,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_max_tokens: Optional[int] = None,
        factor_increase_token_budget: Optional[float] = None,
        max_retries: int = 5,
    ):
        """Initialize LLM wrapper.

        Args:
            requires_think: If True, generate() will raise exception if thinking is None/empty
            shortname: Optional short name for more interpretable log directory names
            print_time: If True, print timing info for each generation
            timeout: Optional timeout in seconds for API calls (default None for no timeout)
            max_tokens: Optional maximum tokens for output (default None)
            reasoning_max_tokens: Optional maximum tokens for thinking/reasoning (default None)
            factor_increase_token_budget: Optional multiplier for token budgets (default None)
                When provided, multiplies both max_tokens and reasoning_max_tokens by this factor
            max_retries: Maximum number of retries on failures (empty output, missing thinking, API errors) (default 3)
        """
        self.requires_think = requires_think
        self.shortname = shortname
        self.print_time = print_time
        self.timeout = timeout
        self.max_retries = max_retries
        self._max_retries_set = False

        # Store base token budgets
        self._base_max_tokens = max_tokens
        self._base_reasoning_max_tokens = reasoning_max_tokens
        self._factor_increase_token_budget: Optional[float] = None

        # Initialize token budgets to base values
        self.max_tokens = max_tokens
        self.reasoning_max_tokens = reasoning_max_tokens

        # Apply factor if provided
        if factor_increase_token_budget is not None:
            self.apply_factor_increase_token_budget(factor_increase_token_budget)

        self.write_to_fs = write_to_filesystem
        if self.write_to_fs:
            self._setup_log_dir()

    @property
    def factor_increase_token_budget(self) -> Optional[float]:
        """Token budget multiplication factor (read-only).

        Use apply_factor_increase_token_budget() to set this value.
        """
        return self._factor_increase_token_budget

    def apply_factor_increase_token_budget(self, factor: float) -> None:
        """Apply a multiplication factor to token budgets.

        Can only be called once. Subsequent calls will raise RuntimeError.

        Args:
            factor: Multiplier to apply to max_tokens and reasoning_max_tokens

        Raises:
            RuntimeError: If called more than once
        """
        # Check if already applied
        if self._factor_increase_token_budget is not None:
            raise RuntimeError(
                f"apply_factor_increase_token_budget() has already been called with factor={self._factor_increase_token_budget}. "
                "This method can only be called once."
            )

        self._factor_increase_token_budget = factor

        # Apply multiplication to token budgets
        old_max_tokens = self.max_tokens
        old_reasoning_max_tokens = self.reasoning_max_tokens
        old_timeout = self.timeout 

        self.max_tokens = round(self._base_max_tokens * factor) if self._base_max_tokens is not None else None
        self.reasoning_max_tokens = round(self._base_reasoning_max_tokens * factor) if self._base_reasoning_max_tokens is not None else None
        self.timeout = (200 if old_timeout is None else old_timeout) * factor

        # Print what we did
        parts = []
        if self._base_max_tokens is not None:
            parts.append(f"max_tokens {self._base_max_tokens}→{self.max_tokens}")
        if self._base_reasoning_max_tokens is not None:
            parts.append(f"reasoning_max_tokens {self._base_reasoning_max_tokens}→{self.reasoning_max_tokens}")

        parts.append(f"timeout {old_timeout}→{self.timeout}")

        if parts:
            print(f"🔢 {self.__class__.__name__}: Applying {factor}x token budget factor: {', '.join(parts)}")

        # Log the update to config
        update_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "apply_factor_increase_token_budget",
            "factor": factor,
            "old_max_tokens": old_max_tokens,
            "new_max_tokens": self.max_tokens,
            "old_reasoning_max_tokens": old_reasoning_max_tokens,
            "new_reasoning_max_tokens": self.reasoning_max_tokens,
            "old_timeout": old_timeout,
            "new_timeout": self.timeout,
        }

        if self.write_to_fs:
            with open(self.config_file, "a") as f:
                f.write(json.dumps(update_entry) + "\n")

    def set_max_retries(self, num_retries: int) -> None:
        """Set the maximum number of retries for failed API calls.

        Can only be called once. Subsequent calls will raise RuntimeError.

        Args:
            num_retries: Number of retries to set

        Raises:
            RuntimeError: If called more than once
        """
        # Check if already set
        if self._max_retries_set:
            raise RuntimeError(
                f"set_max_retries() has already been called with max_retries={self.max_retries}. "
                "This method can only be called once."
            )

        self._max_retries_set = True
        old_max_retries = self.max_retries
        self.max_retries = num_retries

        # Print what we did
        print(f"🔄 {self.__class__.__name__}: Setting max_retries to {num_retries} (was {old_max_retries})")

        # Log the update to config
        update_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "set_max_retries",
            "old_max_retries": old_max_retries,
            "new_max_retries": self.max_retries,
        }

        if self.write_to_fs:
            with open(self.config_file, "a") as f:
                f.write(json.dumps(update_entry) + "\n")

    def _setup_log_dir(self) -> None:
        """Setup logging directory for this LLM instance."""
        now = datetime.now()
        month = now.strftime("%m")
        day = now.strftime("%d")
        hr_min = now.strftime("%H_%M")

        shortname_part = f"{self.shortname}_{hr_min}" if self.shortname else hr_min

        log_dir = Path(f"logs/LLMs/{self.__class__.__name__}/{month}/{day}/{shortname_part}")
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = log_dir
        self.log_file = log_dir / "queries.jsonl"
        self.config_file = log_dir / "config.jsonl"

    def log_config(self, extra_config: Optional[Dict[str, Any]] = None) -> None:
        """Log LLM configuration to config.jsonl in the log directory.

        Should be called at the end of subclass __init__ after setting model and client.
        Logs all init parameters plus model and base_url if available.
        Appends to jsonl file with timestamp and event type.

        Args:
            extra_config: Optional dict of additional config fields from subclasses
        """
        config = {
            "timestamp": datetime.now().isoformat(),
            "event": "init",
            "class": self.__class__.__name__,
            "requires_think": self.requires_think,
            "shortname": self.shortname,
            "print_time": self.print_time,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "reasoning_max_tokens": self.reasoning_max_tokens,
            "_base_max_tokens": self._base_max_tokens,
            "_base_reasoning_max_tokens": self._base_reasoning_max_tokens,
            "factor_increase_token_budget": self.factor_increase_token_budget,
            "max_retries": self.max_retries,
        }

        # Add model if available
        if hasattr(self, 'model'):
            config["model"] = self.model

        # Add base_url if available from client
        if hasattr(self, 'client') and hasattr(self.client, 'base_url'):
            config["base_url"] = str(self.client.base_url)

        # Merge extra_config from subclasses
        if extra_config:
            config.update(extra_config)

        # Append to jsonl file
        if self.write_to_fs:
            with open(self.config_file, "a") as f:
                f.write(json.dumps(config) + "\n")

    def get_log_dir(self) -> str:
        """Return the logging directory path for this LLM instance."""
        return str(self.log_dir)

    async def generate(self, prompts: List[str]) -> List[LLMResponse]:
        """Generate responses for a list of prompts in parallel.

        Args:
            prompts: List of prompt strings

        Returns:
            List of LLMResponse objects, one per prompt

        Raises:
            ValueError: If requires_think=True and any response has None/empty thinking
        """
        # Run async generation
        responses = await asyncio.gather(
            *[self._generate_single_with_logging(i, prompt) for i, prompt in enumerate(prompts)]
        )

        # Validate thinking if required
        if self.requires_think:
            for i, response in enumerate(responses):
                if response.thinking is None or response.thinking == "":
                    print(
                        f"❌ DEBUG: LLM {self.__class__.__name__} requires thinking but received "
                        f"None/empty thinking for prompt {i}"
                    )

        # Log all queries
        self._log_queries(prompts, responses)

        return responses

    async def _generate_single_with_logging(self, prompt_idx: int, prompt: str) -> LLMResponse:
        """Wrapper around _generate_single that adds timing prints and retry logic.

        Args:
            prompt_idx: Index of the prompt
            prompt: Prompt string

        Returns:
            LLMResponse
        """
        # Retry loop for empty outputs, missing required thinking, or API errors
        for attempt in range(1, self.max_retries + 1):
            response = await self._generate_single(prompt)

            # Check if output is empty
            shortname_prefix = f"{self.shortname}_" if self.shortname else ""
            model_name = self.__class__.__name__

            if len(response.output) == 0:
                if attempt < self.max_retries:
                    thinking_len = len(response.thinking) if response.thinking else 0
                    print(
                        f"⚠️  [P{prompt_idx}] {shortname_prefix}{model_name}: Empty output (attempt {attempt}/{self.max_retries})\n"
                        f"    output_len=0, thinking_len={thinking_len} max_tokens={self.max_tokens}, error={response.error!r}, time={getattr(response, 'time_to_respond', None)}s\n"
                        "    Retrying..."
                    )
                    await asyncio.sleep(2)
                    continue
                else:
                    thinking_len = len(response.thinking) if response.thinking else 0
                    print(
                        f"❌ [P{prompt_idx}] {shortname_prefix}{model_name}: Empty output after {self.max_retries} attempts — giving up\n"
                        f"    output_len=0, thinking_len={thinking_len}, error={response.error!r}, time={getattr(response, 'time_to_respond', None)}s"
                    )

            # Check if thinking is required but missing
            if self.requires_think and (response.thinking is None or response.thinking == ""):
                if attempt < self.max_retries:
                    print(
                        f"⚠️  [P{prompt_idx}] {shortname_prefix}{model_name}: Required thinking missing (attempt {attempt}/{self.max_retries})\n"
                        f"    output_len={len(response.output) if response.output else 0}, error={response.error!r}, time={getattr(response, 'time_to_respond', None)}s\n"
                        "    Retrying..."
                    )
                    await asyncio.sleep(2)
                    continue
                else:
                    print(
                        f"❌ [P{prompt_idx}] {shortname_prefix}{model_name}: Required thinking missing after {self.max_retries} attempts — giving up\n"
                        f"    output_len={len(response.output) if response.output else 0}, error={response.error!r}, time={getattr(response, 'time_to_respond', None)}s"
                    )

            # Check if there was an API error
            if response.error is not None:
                if attempt < self.max_retries:
                    print(
                        f"⚠️  [P{prompt_idx}] {shortname_prefix}{model_name}: API error on attempt {attempt}/{self.max_retries}\n"
                        f"    error={response.error!r}, output_len={len(response.output) if response.output else 0}, thinking_len={len(response.thinking) if response.thinking else 0}, time={getattr(response, 'time_to_respond', None)}s\n"
                        "    Retrying..."
                    )
                    await asyncio.sleep(2)
                    continue
                else:
                    print(
                        f"❌ [P{prompt_idx}] {shortname_prefix}{model_name}: API error after {self.max_retries} attempts — giving up\n"
                        f"    error={response.error!r}, output_len={len(response.output) if response.output else 0}, thinking_len={len(response.thinking) if response.thinking else 0}, time={getattr(response, 'time_to_respond', None)}s"
                    )

            # Success or final attempt - break and return
            break

        if self.print_time:
            model_name = self.__class__.__name__
            shortname_prefix = f"{self.shortname}_" if self.shortname else ""
            output_chars = len(response.output)
            time_sec = response.time_to_respond

            print(f"[P{prompt_idx}] {shortname_prefix}{model_name}: finished {output_chars} output chars and {len(response.thinking) if response.thinking else '[0 thinking tokens available]'} thinking chars in {time_sec:.1f}s")

        return response

    @abstractmethod
    async def _generate_single(self, prompt: str) -> LLMResponse:
        """Generate a response for a single prompt.

        Must be implemented by subclasses to handle specific LLM APIs.

        Args:
            prompt: Single prompt string

        Returns:
            LLMResponse with thinking, output, and time_to_respond
        """
        pass

    def _log_queries(self, prompts: List[str], responses: List[LLMResponse]) -> None:
        """Log all queries and responses to jsonl file.

        Args:
            prompts: List of input prompts
            responses: List of LLMResponse objects
        """
        timestamp = datetime.now().isoformat()

        for response in responses:
            # Convert LLMResponse dataclass to dict
            log_entry = asdict(response)
            # Add timestamp
            log_entry["timestamp"] = timestamp

            if self.write_to_fs:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

    async def generate_with_prefil(self, *args, **kwargs) -> LLMResponse:
        """Generate with prefilled assistant response (not implemented yet)."""
        raise NotImplementedError("generate_with_prefil not implemented yet")

