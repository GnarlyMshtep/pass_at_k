"""Base class for SFT evaluation.

Each eval type subclasses SFTEval and defines its own __init__ kwargs
(passed via eval_kwargs in EvalConfig).
"""

from abc import ABC, abstractmethod
from typing import Any


class SFTEval(ABC):
    """Abstract base class for SFT evaluations.

    Subclasses receive eval-specific kwargs in __init__ and must implement run().
    """

    @abstractmethod
    async def run(
        self,
        model: Any,
        tokenizer: Any,
        run_dir: str,
    ) -> list[dict[str, Any]]:
        """Run evaluation on the given model.

        Args:
            model: The trained (merged) model.
            tokenizer: The tokenizer.
            run_dir: Path to the run directory for saving artifacts.

        Returns:
            List of result dicts (one per sample), each containing all
            metrics/values from the evaluation.
        """
        ...
