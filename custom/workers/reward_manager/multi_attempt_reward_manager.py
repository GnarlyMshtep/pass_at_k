"""
Custom reward manager that applies multi-attempt reward processing.
"""
import torch
import numpy as np
from typing import Any, Dict, List
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager import register
from custom.multi_attempt.config import MultiAttemptConfig


@register("multi_attempt_reward_manager")
class MultiAttemptRewardManager(AbstractRewardManager):
    """
    Reward manager that applies multi-attempt reward processing.
    
    This manager wraps another reward manager and applies the multi-attempt
    exploration formula to the computed rewards.
    """
    
    def __init__(
        self,
        base_reward_manager: str = "dapo",
        enabled: bool = True,
        max_attempts: int | None = None,
        num_samples_per_attempt: int | None = None,
        val_max_attempts: int | None = None,
        val_num_samples_per_attempt: int | None = None,
        attempt_template: str | None = None,
        **kwargs
    ):
        """
        Initialize the multi-attempt reward manager.
        
        Args:
            base_reward_manager: Name of the underlying reward manager to wrap
            enabled: Whether to enable multi-attempt processing
            **kwargs: Additional arguments passed to base reward manager
        """
        super().__init__(**kwargs)
        # Track split: 0=train, 1=val in our PPO loaders
        self.num_examine = kwargs.get("num_examine", 0)
        
        # Create the base reward manager
        from verl.workers.reward_manager import get_reward_manager_cls
        base_reward_manager_cls = get_reward_manager_cls(base_reward_manager)
        self.base_reward_manager = base_reward_manager_cls(**kwargs)
        
        # Create multi-attempt configuration from passed parameters (preferred over inferring)
        mac_kwargs = {}
        if max_attempts is not None:
            mac_kwargs["max_attempts"] = int(max_attempts)
        if num_samples_per_attempt is not None:
            mac_kwargs["num_samples_per_attempt"] = int(num_samples_per_attempt)
        if val_max_attempts is not None:
            mac_kwargs["val_max_attempts"] = int(val_max_attempts)
        if val_num_samples_per_attempt is not None:
            mac_kwargs["val_num_samples_per_attempt"] = int(val_num_samples_per_attempt)
        if attempt_template is not None:
            mac_kwargs["attempt_template"] = attempt_template

        self.multi_attempt_config = MultiAttemptConfig(
            enabled=enabled,
            **mac_kwargs,
        )
    
    def __call__(self, batch: DataProto, return_dict: bool = False, **kwargs):
        """
        Compute rewards using the base reward manager and apply multi-attempt processing.
        
        Args:
            batch: Input batch with generated responses
            return_dict: Whether to return additional information
            **kwargs: Additional arguments
            
        Returns:
            Processed rewards with multi-attempt formula applied
        """
        # Always call base manager with return_dict=True to obtain extra info for MA computation
        base_result = self.base_reward_manager(batch, return_dict=True, **kwargs)
        base_reward_tensor: torch.Tensor = base_result["reward_tensor"]
        reward_extra_info: Dict[str, List[Any]] = base_result.get("reward_extra_info", {})

        if self.multi_attempt_config.enabled:
            new_reward_tensor, updated_extra_info, extra_reward_metrics = self._apply_multi_attempt_from_extra_info(
                batch=batch,
                base_reward_tensor=base_reward_tensor,
                reward_extra_info=reward_extra_info,
            )
        else:
            new_reward_tensor = base_reward_tensor
            updated_extra_info = reward_extra_info
            extra_reward_metrics = {}

        if return_dict:
            return {
                "reward_tensor": new_reward_tensor,
                "reward_extra_info": updated_extra_info,
                "extra_reward_metrics": extra_reward_metrics,
            }
        else:
            return new_reward_tensor
    
    def verify(self, batch: DataProto, **kwargs):
        """
        Verify rewards (delegate to base reward manager).
        
        Args:
            batch: Input batch
            **kwargs: Additional arguments
            
        Returns:
            Verification results from base reward manager
        """
        return self.base_reward_manager.verify(batch, **kwargs)
    
    def _apply_multi_attempt_from_extra_info(
        self,
        batch: DataProto,
        base_reward_tensor: torch.Tensor,
        reward_extra_info: Dict[str, List[Any]],
    ) -> tuple[torch.Tensor, Dict[str, List[Any]], Dict[str, float]]:
        """
        Compute multi-attempt outcome rewards from correctness flags, add format bonus, and
        place the final score on the last token (Dapo convention). Also attach extra info and
        aggregated metrics.
        """
        attempt_ids = batch.non_tensor_batch.get("attempt_id", None)
        sample_ids = batch.non_tensor_batch.get("sample_id", None)
        uids = batch.non_tensor_batch.get("uid", None)

        if attempt_ids is None or uids is None:
            return base_reward_tensor, reward_extra_info, {}

        # Determine parameters for current split
        if self.num_examine == 1:
            k = int(self.multi_attempt_config.val_max_attempts)
            N = int(self.multi_attempt_config.val_num_samples_per_attempt)
        else:
            k = int(self.multi_attempt_config.max_attempts)
            N = int(self.multi_attempt_config.num_samples_per_attempt)

        if k <= 0 or N <= 0:
            return base_reward_tensor, reward_extra_info, {}

        # Fetch correctness and format from extra info; fallback to zeros if missing
        is_correct_list = reward_extra_info.get("is_correct", None)
        format_score_list = reward_extra_info.get("format_score", None)

        bsz = base_reward_tensor.shape[0]
        if is_correct_list is None or len(is_correct_list) != bsz:
            # Fallback: derive correctness from base sequence reward >= 1.0
            seq_rewards = base_reward_tensor.sum(dim=-1)
            is_correct = (seq_rewards >= 1.0).to(dtype=torch.float32)
        else:
            is_correct = torch.tensor(is_correct_list, dtype=torch.float32, device=base_reward_tensor.device)

        if format_score_list is None or len(format_score_list) != bsz:
            format_score = torch.zeros(bsz, dtype=torch.float32, device=base_reward_tensor.device)
        else:
            format_score = torch.tensor(format_score_list, dtype=torch.float32, device=base_reward_tensor.device)

        # Group indices by uid
        from collections import defaultdict
        prompt_groups: Dict[Any, List[int]] = defaultdict(list)
        for i in range(len(uids)):
            prompt_groups[uids[i]].append(i)

        # Prepare outputs
        outcome_rewards = torch.zeros(bsz, dtype=torch.float32, device=base_reward_tensor.device)
        group_pass_vals: List[float] = []
        group_pass_per_sample = torch.zeros(bsz, dtype=torch.float32, device=base_reward_tensor.device)

        for prompt_id, sample_indices in prompt_groups.items():
            group_attempt_ids = [int(attempt_ids[i]) for i in sample_indices]
            # Count correct samples per attempt
            correct_counts = np.zeros(k, dtype=np.int32)
            for idx_within_group, abs_idx in enumerate(sample_indices):
                a_id = group_attempt_ids[idx_within_group]
                if 0 <= a_id < k and is_correct[abs_idx] >= 1.0:
                    correct_counts[a_id] += 1

            # Group-level pass@k = 1 - Π_r ((N - C_r)/N)
            product_all = 1.0
            for r in range(k):
                product_all *= (float(N - correct_counts[r]) / float(N))
            group_pass = 1.0 - product_all
            group_pass_vals.append(group_pass)

            # Per-sample outcome reward
            for idx_within_group, abs_idx in enumerate(sample_indices):
                a_id = group_attempt_ids[idx_within_group]
                if is_correct[abs_idx] >= 1.0:
                    new_outcome = 1.0
                else:
                    # Incorrect: exclude its attempt from the product
                    if 0 <= a_id < k:
                        new_outcome = self._compute_incorrect_reward(a_id, correct_counts, k, N)
                    else:
                        new_outcome = 0.0
                outcome_rewards[abs_idx] = float(new_outcome)
                group_pass_per_sample[abs_idx] = float(group_pass)
        if outcome_rewards.max() > 1.0 or outcome_rewards.min() < 0:
            breakpoint()
        # Final per-sample reward = outcome + format
        final_scores = outcome_rewards + format_score

        # Map to token-level, last-token only
        last_token_indices = self._get_last_token_indices(batch, base_reward_tensor)
        new_token_rewards = torch.zeros_like(base_reward_tensor)
        for i in range(bsz):
            j = last_token_indices[i]
            if j >= 0:
                new_token_rewards[i, j] = final_scores[i]

        # Update extra info with per-sample fields
        updated_extra = dict(reward_extra_info)
        updated_extra.setdefault("group_pass_at_k", [])
        updated_extra.setdefault("in_group_reward", [])
        updated_extra.setdefault("final_reward", [])
        updated_extra.setdefault("attempt_id", [])
        updated_extra.setdefault("sample_id", [])
        # Ensure attempt_id/sample_id arrays
        attempt_ids_arr = attempt_ids if isinstance(attempt_ids, np.ndarray) else np.asarray(attempt_ids)
        sample_ids_arr = sample_ids if isinstance(sample_ids, np.ndarray) else (
            np.asarray(sample_ids) if sample_ids is not None else np.zeros(bsz, dtype=np.int32)
        )
        for i in range(bsz):
            updated_extra["group_pass_at_k"].append(float(group_pass_per_sample[i].item()))
            updated_extra["in_group_reward"].append(float(outcome_rewards[i].item()))
            updated_extra["final_reward"].append(float(final_scores[i].item()))
            updated_extra["attempt_id"].append(int(attempt_ids_arr[i]))
            updated_extra["sample_id"].append(int(sample_ids_arr[i]))

        # Aggregated metrics
        extra_reward_metrics = {}
        if len(group_pass_vals) > 0:
            extra_reward_metrics[f"multi_attempt/pass@{k}"] = float(np.mean(group_pass_vals))

        return new_token_rewards, updated_extra, extra_reward_metrics
    
    def _compute_incorrect_reward(self, attempt_id: int, correct_counts: np.ndarray, max_attempts: int, num_samples_per_attempt: int) -> float:
        """
        Compute reward for incorrect completions using the formula:
        1 - ∏(r≠i) (N-C_r)/N^(k-1)
        """
        N = num_samples_per_attempt
        k = max_attempts
        
        # Compute the product term: ∏(r≠i) (N-C_r)
        product_term = 1.0
        for r in range(k):
            if r != attempt_id:
                C_r = correct_counts[r]
                product_term *= ((N - C_r) / N)
        
        # Apply the formula: 1 - ∏(r≠i) (N-C_r)/N^(k-1)
        reward = 1.0 - (product_term)
        
        return reward

    def _get_last_token_indices(self, batch: DataProto, base_reward_tensor: torch.Tensor) -> List[int]:
        """
        Determine last response token index for each sample. Prefer using nonzero in base_reward_tensor;
        fallback to attention_mask lengths.
        """
        bsz, resp_len = base_reward_tensor.shape
        indices: List[int] = [-1] * bsz
        try:
            nz = (base_reward_tensor != 0).int()
            # pick the last nonzero index per row if exists
            for i in range(bsz):
                nz_positions = torch.nonzero(nz[i], as_tuple=False)
                if nz_positions.numel() > 0:
                    indices[i] = int(nz_positions[-1, 0].item())
        except Exception:
            pass

        # Fallback using attention_mask
        try:
            resp_max_len = batch.batch["responses"].shape[-1]
            response_mask = batch.batch["attention_mask"][:, -resp_max_len:]
            resp_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)
            for i in range(bsz):
                if indices[i] < 0:
                    L = int(resp_lengths[i].item())
                    if L > 0:
                        indices[i] = L - 1
                    else:
                        indices[i] = 0
        except Exception:
            # If all else fails, set to last position
            for i in range(bsz):
                if indices[i] < 0:
                    indices[i] = resp_len - 1

        return indices
