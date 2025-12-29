"""
Prompt processing utilities for multi-attempt exploration strategy.
"""
from typing import List, Dict, Any
import numpy as np
from verl.protocol import DataProto
from .config import MultiAttemptConfig


class MultiAttemptPromptProcessor:
    """Handles prompt processing for multi-attempt exploration strategy."""
    
    def __init__(self, config: MultiAttemptConfig):
        self.config = config
    
    def _unescape_cli_string(self, text: str) -> str:
        """Convert common escape sequences in CLI-passed strings to real chars.
        Only handles simple sequences (\\n, \\t, \\r) to avoid over-decoding.
        """
        if not isinstance(text, str):
            return text
        return text.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    
    def add_attempt_tokens_to_prompts(self, prompts: List[str], attempt_id: int) -> List[str]:
        """
        Add attempt-specific tokens to prompts.
        
        Args:
            prompts: List of original prompts
            attempt_id: The attempt ID to append
            
        Returns:
            List of prompts with attempt tokens appended
        """
        # Support both {attempt_id} and optional {max_attempts} placeholders.
        # Interpret escaped sequences (e.g., "\\n") from CLI overrides.
        template = self._unescape_cli_string(self.config.attempt_template)
        attempt_text = template.format(
            attempt_id=attempt_id + 1,
            max_attempts=self.config.max_attempts,
        )
        return [prompt + attempt_text for prompt in prompts]
    
    def create_structured_batch(self, original_batch: DataProto) -> DataProto:
        """
        Create a structured batch with attempt_id and sample_id for multi-attempt strategy.
        
        Args:
            original_batch: Original batch from dataset
            
        Returns:
            Structured batch with attempt and sample IDs
        """
        if not self.config.enabled:
            return original_batch
        
        batch_size = len(original_batch)
        total_samples = batch_size * self.config.total_rollouts_per_prompt
        
        # Create attempt_id and sample_id arrays
        attempt_ids = np.zeros(total_samples, dtype=np.int32)
        sample_ids = np.zeros(total_samples, dtype=np.int32)
        
        # Fill attempt_id and sample_id arrays
        for i in range(batch_size):
            for attempt_id in range(self.config.max_attempts):
                for sample_id in range(self.config.num_samples_per_attempt):
                    idx = i * self.config.total_rollouts_per_prompt + attempt_id * self.config.num_samples_per_attempt + sample_id
                    attempt_ids[idx] = attempt_id
                    sample_ids[idx] = sample_id
        
        # Create new batch with repeated data
        structured_batch = original_batch.repeat(
            repeat_times=self.config.total_rollouts_per_prompt, 
            interleave=True
        )
        
        # Add attempt_id and sample_id to non_tensor_batch
        structured_batch.non_tensor_batch["attempt_id"] = attempt_ids
        structured_batch.non_tensor_batch["sample_id"] = sample_ids
        
        return structured_batch
    
    def process_prompts_with_attempts(self, batch: DataProto) -> DataProto:
        """
        Process prompts by adding attempt-specific tokens.
        
        Args:
            batch: Batch with attempt_id and sample_id already added
            
        Returns:
            Batch with modified prompts containing attempt tokens
        """
        if not self.config.enabled:
            return batch
        
        # Get raw prompts from the batch
        raw_prompts = batch.non_tensor_batch.get("raw_prompt", [])
        if not raw_prompts:
            return batch
        
        attempt_ids = batch.non_tensor_batch["attempt_id"]
        processed_prompts = []
        
        # Process each prompt with its corresponding attempt_id
        for i, (prompt, attempt_id) in enumerate(zip(raw_prompts, attempt_ids)):
            processed_prompt = self.add_attempt_tokens_to_prompts([prompt], attempt_id)[0]
            processed_prompts.append(processed_prompt)
        
        # Update the batch with processed prompts
        batch.non_tensor_batch["raw_prompt"] = np.array(processed_prompts, dtype=object)
        
        return batch
