"""
Custom dataset class for math problems that appends instruction suffix to questions.
"""
import os
from verl.utils.dataset.rl_dataset import RLHFDataset


class CustomMathDataset(RLHFDataset):
    """
    Custom RLHF dataset that appends a suffix to user messages.
    
    This is useful for adding instructions like "Please reason step by step, 
    and put your final answer within \\boxed{}" to math problems.
    """
    
    def _build_messages(self, example: dict):
        """
        Build messages from example and append suffix to the last user message.
        
        The suffix can be configured via config.prompt_suffix, or defaults to:
        "\\nPlease reason step by step, and put your final answer within \\boxed{}"
        """
        messages = super()._build_messages(example)
        
        # Handle case where messages might be a string or list of strings
        # Convert to proper message format if needed
        if isinstance(messages, str):
            # If it's a string, convert to a list with a user message
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            # Check if first item is a string (not a dict)
            raise ValueError(f"messages is a list of strings: {messages}")
        # Get suffix from config, or use default
        prompt_suffix = self.config["prompt_suffix"]
        if prompt_suffix is None:
            raise ValueError("prompt_suffix is required")
        
        # Append suffix to the last user message
        for msg in reversed(messages):
            # Ensure msg is a dict before accessing keys
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                # Handle both string content and list content (for multimodal)
                if isinstance(msg.get("content"), str):
                    msg["content"] += prompt_suffix
                else:
                    raise ValueError(f"content is not a string: {msg.get('content')}")
                break
        
        return messages
    

    
    def __getitem__(self, item):
        """
        Override __getitem__ to ensure reward_model and data_source are present.
        If these fields are missing from the parquet file, construct them.
        """
        row_dict = super().__getitem__(item)
        
        # Add reward_model if missing (usually comes from parquet file, but some datasets don't have it)
        if "reward_model" not in row_dict or row_dict.get("reward_model") is None:
            # Try to construct from extra_info or use default
            ground_truth = None
            extra_info = row_dict.get("extra_info", {})
            if isinstance(extra_info, dict):
                # Try to get ground_truth from extra_info
                ground_truth = extra_info.get("ground_truth") or extra_info.get("answer") or extra_info.get("solution")
            if ground_truth is None: 
                ground_truth = row_dict.get("answer", None)
            if ground_truth is None:
                ground_truth = row_dict.get("solution", None)
            if ground_truth is None:
                ground_truth = row_dict.get("ground_truth", None)
            if ground_truth is None:
                ground_truth = row_dict.get("target", None)
            if ground_truth is None:
                raise ValueError("ground_truth is required")
            row_dict["reward_model"] = {
                "style": "rule",
                "ground_truth": ground_truth,
            }
        assert "data_source" in row_dict, "data_source is required"
        return row_dict

