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
    
    def _get_dataset_name(self):
        """
        Extract dataset name from config or file paths.
        Tries in order:
        1. config.dataset_name
        2. config.data_source
        3. Extract from file path (e.g., /data/math12k/data/train.parquet -> math12k)
        4. Default: "unknown"
        """
        # Try config first
        dataset_name = self.config.get("dataset_name") or self.config.get("data_source")
        if dataset_name:
            return str(dataset_name)
        
        # Try to extract from file path
        if self.data_files:
            # Convert to regular list if it's a ListConfig or similar
            try:
                from omegaconf import ListConfig
                if isinstance(self.data_files, ListConfig):
                    file_paths = [str(f) for f in self.data_files]
                elif isinstance(self.data_files, list):
                    file_paths = [str(f) for f in self.data_files]
                else:
                    file_paths = [str(self.data_files)]
            except (ImportError, TypeError):
                # Fallback if omegaconf not available or conversion fails
                if isinstance(self.data_files, list):
                    file_paths = [str(f) for f in self.data_files]
                else:
                    file_paths = [str(self.data_files)]
            
            if file_paths:
                # Use the first file path
                file_path = file_paths[0]
                # Extract directory name from path like /data/math12k/data/train.parquet
                # or /data/aime24/data/train.parquet
                path_parts = file_path.split(os.sep)
                # Look for common patterns: .../data/<dataset_name>/data/...
                for i, part in enumerate(path_parts):
                    if part == "data" and i + 1 < len(path_parts):
                        dataset_name = path_parts[i + 1]
                        if dataset_name and dataset_name != "data":  # Avoid picking up nested "data" dirs
                            return dataset_name
        
        return "unknown"
    
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
                ground_truth = row_dict.get("answer")
            if ground_truth is None:
                ground_truth = row_dict.get("solution")
            if ground_truth is None:
                ground_truth = row_dict.get("ground_truth")
            if ground_truth is None:
                ground_truth = row_dict.get("target")
            if ground_truth is None:
                raise ValueError("ground_truth is required")
            row_dict["reward_model"] = {
                "style": "rule",
                "ground_truth": ground_truth,
            }
        
        # Add data_source if missing (required by reward manager)
        if "data_source" not in row_dict or row_dict.get("data_source") is None:
            row_dict["data_source"] = self._get_dataset_name()
        
        return row_dict

