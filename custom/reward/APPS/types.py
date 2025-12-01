from dataclasses import dataclass
from typing import Any , Optional
from abc import ABC, abstractmethod

@dataclass
class AbstractGeneratedSample(ABC):
    """Base class for all generated samples across tasks."""
    response: str


@dataclass
class ScoredSampleBase(ABC):
    """Base class for all scored samples across tasks.

    Convention: Each subclass should include a task-specific reward field named
    final_{ClassName}_reward that stores the reward computed by that task's
    compute_final_reward method. For example:
    - APPSScoredSample has final_APPSMainBase_reward
    - APPSBackdoorScoredSample has final_APPSBackdoorScoredSample_reward

    Each subclass must implement get_final_reward() to return its task-specific reward.
    """
    generation: AbstractGeneratedSample

    @abstractmethod
    def get_final_reward(self) -> float:
        """Return the task-specific final reward.

        Each subclass implements this to return its uniquely-named reward field.
        For example:
        - APPSScoredSample returns self.final_APPSMainBase_reward
        - APPSBackdoorScoredSample returns self.final_APPSBackdoorScoredSample_reward
        """
        raise NotImplementedError


@dataclass
class APPSQuestion:
    """Represents a filtered APPS question with all necessary metadata."""
    problem_id: str
    question: str  # Problem description
    input_output: str  # JSON string with {"inputs": [...], "outputs": [...]}   
    known_good_solution: str  # A solution that passes all test cases
    num_test_cases: int
    validation_script: str  # Python script that validates inputs, or "NOCONSTRAINTS"
    original_apps_problem: dict  # All fields from the original APPS dataset item
    split: str  # Which split this question belongs to
    full_prompt: list # the full input prompt, usually including "system prompt" and "user" fields


@dataclass
class APPSGeneratedSample(AbstractGeneratedSample):
    """Represents a generated code sample for an APPS question."""
    question: APPSQuestion
    response: str  



@dataclass
class APPSScoredSample(ScoredSampleBase):
    """Represents a scored APPS sample with execution results."""
    # Inherited: generation (AbstractGeneratedSample)
    tests_passed: int
    tests_total: int
    frac_test_cases_passing: float  # Normalized score (tests_passed / tests_total)
    final_APPSMainBase_reward: float  # Final reward for APPSMainBase (equals frac_test_cases_passing)
    execution_results: dict  # Full results from verify_solution
    error: Any
    full_sample: APPSGeneratedSample
    extracted_code:Optional[str]

    def get_final_reward(self) -> float:
        """Return the APPSMainBase task-specific reward."""
        return self.final_APPSMainBase_reward
