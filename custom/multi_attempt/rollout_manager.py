"""
Rollout management for multi-attempt exploration strategy.
"""
import numpy as np
from verl.protocol import DataProto
from .config import MultiAttemptConfig
from .prompt_processor import MultiAttemptPromptProcessor


class MultiAttemptRolloutManager:
    """Manages rollout logic for multi-attempt exploration strategy."""
    
    def __init__(self, config: MultiAttemptConfig):
        self.config = config
        self.prompt_processor = MultiAttemptPromptProcessor(config)
    
    def prepare_generation_batch(self, original_batch: DataProto) -> DataProto:
        """
        Prepare batch for generation with multi-attempt structure.
        
        Args:
            original_batch: Original batch from dataset
            
        Returns:
            Batch prepared for generation with attempt structure
        """
        if not self.config.enabled:
            return original_batch
        
        # Create structured batch with attempt_id and sample_id
        structured_batch = self.prompt_processor.create_structured_batch(original_batch)
        
        # Process prompts with attempt tokens
        generation_batch = self.prompt_processor.process_prompts_with_attempts(structured_batch)
        
        return generation_batch
    
    def prepare_training_batch(self, original_batch: DataProto, generation_output: DataProto) -> DataProto:
        """
        Prepare batch for training by combining original batch with generation output.
        
        Args:
            original_batch: Original batch from dataset
            generation_output: Output from generation step
            
        Returns:
            Combined batch ready for training
        """
        if not self.config.enabled:
            # Standard behavior: repeat original batch to match generation output
            repeated_batch = original_batch.repeat(
                repeat_times=self.config.total_rollouts_per_prompt,
                interleave=True
            )
            return repeated_batch.union(generation_output)
        
        # For multi-attempt, we need to ensure proper alignment
        # The generation_output should already have the correct structure
        # We just need to add the original batch data
        
        # Create structured original batch
        structured_original = self.prompt_processor.create_structured_batch(original_batch)
        
        # Combine with generation output
        training_batch = structured_original.union(generation_output)
        
        return training_batch
    
    def create_uid_mapping(self, batch_size: int) -> np.ndarray:
        """
        Create UID mapping for proper grouping in advantage computation.
        
        Args:
            batch_size: Size of the original batch
            
        Returns:
            Array of UIDs for grouping
        """
        if not self.config.enabled:
            # Standard behavior: each sample gets a unique UID
            return np.array([f"uid_{i}" for i in range(batch_size)], dtype=object)
        
        # For multi-attempt: group by original prompt
        uids = []
        for i in range(batch_size):
            for _ in range(self.config.total_rollouts_per_prompt):
                uids.append(f"prompt_{i}")
        
        return np.array(uids, dtype=object)
    
    def get_rollout_parameters(self) -> dict:
        """
        Get rollout parameters for the training script.
        
        Returns:
            Dictionary with rollout parameters
        """
        if not self.config.enabled:
            return {"rollout.n": 8}  # Default value
        
        return {
            "rollout.n": self.config.total_rollouts_per_prompt,
            "max_attempts": self.config.max_attempts,
            "num_samples_per_attempt": self.config.num_samples_per_attempt
        }
