"""
Configuration for multi-attempt exploration strategy.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MultiAttemptConfig:
    """Configuration for multi-attempt exploration strategy."""
    
    # Core parameters (used for prompt processing)
    max_attempts: int = 3
    num_samples_per_attempt: int = 4
    
    # Prompt template for attempt-specific text
    attempt_template: str = "\n<attempt-{attempt_id}>"
    
    # Whether to enable multi-attempt strategy
    enabled: bool = True
    
    # Validation parameters (can be different from training)
    val_max_attempts: int = 3
    val_num_samples_per_attempt: int = 4
    
    # Advantage computation parameters
    epsilon: float = 1e-6

    # COMA-lite credit assignment hyperparameters
    # Smoothing for per-attempt success rate p_hat
    alpha: float = 1.0
    # Max weight for self-centered term
    w_max: float = 5.0
    # Coefficient for the counterfactual marginal credit term
    lambda_cf: float = 0.5

    # Advantage mixing hyperparameters
    # Mixing weights for self and group streams
    w1: float = 1.0
    w2: float = 0.5
    # Clipping bound for mixed advantage placed on last token
    a_max: float = 3.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.max_attempts > 0, "max_attempts must be positive"
        assert self.num_samples_per_attempt > 0, "num_samples_per_attempt must be positive"
        assert self.val_max_attempts > 0, "val_max_attempts must be positive"
        assert self.val_num_samples_per_attempt > 0, "val_num_samples_per_attempt must be positive"
        # Require {attempt_id} placeholder; {max_attempts} is optional but supported.
        assert "{attempt_id}" in self.attempt_template, "attempt_template must contain {attempt_id} placeholder"
    
    @property
    def total_rollouts_per_prompt(self) -> int:
        """Total number of rollouts per original prompt for training."""
        return self.max_attempts * self.num_samples_per_attempt
    
    @property
    def val_total_rollouts_per_prompt(self) -> int:
        """Total number of rollouts per original prompt for validation."""
        return self.val_max_attempts * self.val_num_samples_per_attempt
