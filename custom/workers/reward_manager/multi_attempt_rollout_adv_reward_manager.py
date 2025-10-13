"""
Reward manager that computes rollout-based advantages via random groupings per
prompt, standardizes group rewards, averages across groupings, and writes the
result directly on the last token as the training reward.

This implements the second approach from custom/notebooks/adv_simulation.ipynb
(compute_rollout_advantages), adapted to the online PPO setting.
"""
import random
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager import register
from custom.multi_attempt.config import MultiAttemptConfig


@register("multiattemptgroupbasedadvrewardmanager")
class MultiAttemptRolloutAdvantageRewardManager(AbstractRewardManager):
    def __init__(
        self,
        base_reward_manager: str = "dapo",
        enabled: bool = True,
        max_attempts: int | None = None,
        num_samples_per_attempt: int | None = None,
        val_max_attempts: int | None = None,
        val_num_samples_per_attempt: int | None = None,
        attempt_template: str | None = None,
        attempt_insertion_position: str | None = None,
        num_groupings: int | None = None,
        attempt_id_reward_ratio: float | None = None,
        pass_at_k_weight: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # Track split: 0=train, 1=val in our PPO loaders
        self.num_examine = kwargs.get("num_examine", 0)

        # Base reward manager to obtain correctness and metadata
        from verl.workers.reward_manager import get_reward_manager_cls

        base_reward_manager_cls = get_reward_manager_cls(base_reward_manager)
        self.base_reward_manager = base_reward_manager_cls(**kwargs)

        mac_kwargs: Dict[str, Any] = {}
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
        if attempt_insertion_position is not None:
            mac_kwargs["attempt_insertion_position"] = attempt_insertion_position
        if num_groupings is not None:
            mac_kwargs["num_groupings"] = int(num_groupings)
        if attempt_id_reward_ratio is not None:
            mac_kwargs["attempt_id_reward_ratio"] = float(attempt_id_reward_ratio)
        if pass_at_k_weight is not None:
            mac_kwargs["pass_at_k_weight"] = float(pass_at_k_weight)

        self.multi_attempt_config = MultiAttemptConfig(
            enabled=enabled,
            **mac_kwargs,
        )

    def __call__(self, batch: DataProto, return_dict: bool = False, **kwargs: Any):
        # Query base reward manager for sequence-level reward tensor and extra info
        base_result = self.base_reward_manager(batch, return_dict=True, **kwargs)
        base_reward_tensor: torch.Tensor = base_result["reward_tensor"]
        reward_extra_info: Dict[str, List[Any]] = base_result.get("reward_extra_info", {})

        if self.multi_attempt_config.enabled:
            new_reward_tensor, updated_extra_info = self._apply_rollout_advantages(
                batch=batch,
                base_reward_tensor=base_reward_tensor,
                reward_extra_info=reward_extra_info,
            )
        else:
            new_reward_tensor = base_reward_tensor
            updated_extra_info = reward_extra_info

        if return_dict:
            return {
                "reward_tensor": new_reward_tensor,
                "reward_extra_info": updated_extra_info,
            }
        else:
            return new_reward_tensor

    def verify(self, batch: DataProto, **kwargs: Any):
        return self.base_reward_manager.verify(batch, **kwargs)

    def _apply_rollout_advantages(
        self,
        batch: DataProto,
        base_reward_tensor: torch.Tensor,
        reward_extra_info: Dict[str, List[Any]],
    ) -> tuple[torch.Tensor, Dict[str, List[Any]]]:
        attempt_ids = batch.non_tensor_batch.get("attempt_id", None)
        sample_ids = batch.non_tensor_batch.get("sample_id", None)
        uids = batch.non_tensor_batch.get("uid", None)

        if attempt_ids is None or uids is None:
            return base_reward_tensor, reward_extra_info

        # Number of random groupings to average over
        num_groupings = int(self.multi_attempt_config.num_groupings)

        # Correctness signals; fallback to seq reward >= 1.0 if absent
        bsz = base_reward_tensor.shape[0]
        is_correct_list = reward_extra_info.get("is_correct", None)
        if is_correct_list is None or len(is_correct_list) != bsz:
            seq_rewards = base_reward_tensor.sum(dim=-1)
            is_correct = (seq_rewards >= 1.0).to(dtype=torch.float32)
        else:
            is_correct = torch.tensor(is_correct_list, dtype=torch.float32, device=base_reward_tensor.device)

        # Group sample indices by prompt uid
        prompt_groups: Dict[Any, List[int]] = defaultdict(list)
        for i in range(len(uids)):
            prompt_groups[uids[i]].append(i)

        # Precompute last token indices once
        last_token_indices = self._get_last_token_indices(batch, base_reward_tensor)
        
        # Compute pass@k advantages (rollout-based with mixed attempt groups)
        pass_at_k_advantages = self._compute_pass_at_k_advantages(
            prompt_groups=prompt_groups,
            attempt_ids=attempt_ids,
            is_correct=is_correct,
            num_groupings=num_groupings,
            last_token_indices=last_token_indices,
            base_reward_tensor=base_reward_tensor,
        )
        
        # Compute pass@1 advantages (within-attempt comparison)
        pass_at_1_advantages = self._compute_pass_at_1_advantages(
            batch=batch,
            base_reward_tensor=base_reward_tensor,
            is_correct=is_correct,
            uids=uids,
            attempt_ids=attempt_ids,
            prompt_groups=prompt_groups,
        )
        
        # Combine both advantages using weighted sum
        # Formula: pass_at_k_weight * pass@k + (1 - pass_at_k_weight) * pass@1
        w = float(self.multi_attempt_config.pass_at_k_weight)
        combined_advantages_1d = w * pass_at_k_advantages + (1 - w) * pass_at_1_advantages
        
        # Write combined advantages to last token positions
        mixed_advantages = torch.zeros_like(base_reward_tensor)
        for i in range(bsz):
            j = last_token_indices[i]
            if j >= 0:
                mixed_advantages[i, j] = combined_advantages_1d[i]

        # Prepare extra info fields
        updated_extra = dict(reward_extra_info)
        updated_extra.setdefault("final_reward", [])
        updated_extra.setdefault("attempt_id", [])
        updated_extra.setdefault("sample_id", [])

        attempt_ids_arr = attempt_ids if isinstance(attempt_ids, np.ndarray) else np.asarray(attempt_ids)
        sample_ids_arr = sample_ids if isinstance(sample_ids, np.ndarray) else (
            np.asarray(sample_ids) if sample_ids is not None else np.zeros(bsz, dtype=np.int32)
        )

        for i in range(bsz):
            j = last_token_indices[i]
            val = mixed_advantages[i, j] if j >= 0 else 0.0
            updated_extra["final_reward"].append(float(val.item() if isinstance(val, torch.Tensor) else val))
            updated_extra["attempt_id"].append(int(attempt_ids_arr[i]))
            updated_extra["sample_id"].append(int(sample_ids_arr[i]))

        return mixed_advantages, updated_extra

    def _compute_pass_at_k_advantages(
        self,
        prompt_groups: Dict[Any, List[int]],
        attempt_ids: List[int],
        is_correct: torch.Tensor,
        num_groupings: int,
        last_token_indices: List[int],
        base_reward_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pass@k advantages via random groupings where each group contains
        at most one sample per attempt_id.
        
        Returns:
            Tensor of shape (bsz,) containing per-sample pass@k advantages
        """
        bsz = base_reward_tensor.shape[0]
        ra_accum_all = np.zeros(bsz, dtype=np.float64)
        
        # For each prompt group, compute rollout-averaged advantages
        for _pid, sample_indices in prompt_groups.items():
            if len(sample_indices) == 0:
                continue

            # Extract per-sample attempt ids and correctness for this prompt
            group_attempt_ids = [int(attempt_ids[i]) for i in sample_indices]

            # Build per-group averaged advantages array aligned with sample_indices
            ra_accum = np.zeros(len(sample_indices), dtype=np.float64)

            # Create tuples of (local_index, attempt_id) for grouping
            local_tuples = [(loc, group_attempt_ids[loc]) for loc in range(len(sample_indices))]

            for _ in range(num_groupings):
                # Shuffle and greedily assemble groups with unique attempt membership
                random.shuffle(local_tuples)
                remaining = list(local_tuples)
                groups: List[List[int]] = []  # list of lists of local indices
                while remaining:
                    used_attempts = set()
                    group: List[int] = []
                    next_remaining: List[tuple[int, int]] = []
                    for loc, a_id in remaining:
                        if a_id not in used_attempts:
                            group.append(loc)
                            used_attempts.add(a_id)
                        else:
                            next_remaining.append((loc, a_id))
                    groups.append(group)
                    remaining = next_remaining

                # Compute group rewards and z-score across groups
                g_rewards = np.zeros(len(groups), dtype=np.float64)
                r_base = float(self.multi_attempt_config.attempt_id_reward_ratio)
                for gi, group in enumerate(groups):
                    # group reward depends on earliest attempt id that solved the task
                    min_solving_attempt: int | None = None
                    for loc in group:
                        abs_idx = sample_indices[loc]
                        if is_correct[abs_idx] >= 1.0:
                            a_id = int(group_attempt_ids[loc])
                            if (min_solving_attempt is None) or (a_id < min_solving_attempt):
                                min_solving_attempt = a_id
                    if min_solving_attempt is None:
                        g_rewards[gi] = 0.0
                    else:
                        # reward = r ^ (min solving attempt id)
                        g_rewards[gi] = float(r_base ** float(min_solving_attempt))

                mu = float(np.mean(g_rewards))
                sd = float(np.std(g_rewards))
                if sd <= 1e-12:
                    g_adv = np.zeros_like(g_rewards)
                else:
                    g_adv = (g_rewards - mu) / sd

                # Accumulate standardized advantage to members
                for gi, group in enumerate(groups):
                    adv_val = float(g_adv[gi])
                    for loc in group:
                        ra_accum[loc] += adv_val

            # Average across groupings
            if num_groupings > 0:
                ra_vals = ra_accum / float(num_groupings)
            else:
                ra_vals = ra_accum

            # Store averaged advantages aligned with absolute indices
            for offset, abs_idx in enumerate(sample_indices):
                ra_accum_all[abs_idx] = float(ra_vals[offset])

        return torch.tensor(ra_accum_all, dtype=torch.float32, device=base_reward_tensor.device)

    def _compute_pass_at_1_advantages(
        self,
        batch: DataProto,
        base_reward_tensor: torch.Tensor,
        is_correct: torch.Tensor,
        uids: List[Any],
        attempt_ids: List[int],
        prompt_groups: Dict[Any, List[int]],
    ) -> torch.Tensor:
        """
        Compute pass@1 advantages by comparing samples within the same attempt_id
        for each prompt.
        
        Returns:
            Tensor of shape (bsz,) containing per-sample pass@1 advantages
        """
        bsz = base_reward_tensor.shape[0]
        pass_at_1_advantages = np.zeros(bsz, dtype=np.float64)
        
        # For each prompt, group by attempt_id and compute within-attempt advantages
        for _pid, sample_indices in prompt_groups.items():
            if len(sample_indices) == 0:
                continue
            
            # Group samples by attempt_id within this prompt
            attempt_groups: Dict[int, List[int]] = defaultdict(list)
            for idx in sample_indices:
                a_id = int(attempt_ids[idx])
                attempt_groups[a_id].append(idx)
            
            # Compute advantages within each attempt group
            for _a_id, group_indices in attempt_groups.items():
                if len(group_indices) <= 1:
                    # Single sample or empty group - advantage is 0
                    continue
                
                # Get correctness for this group
                group_rewards = np.array([float(is_correct[i].item()) for i in group_indices], dtype=np.float64)
                
                # Z-score within this attempt group
                mu = float(np.mean(group_rewards))
                sd = float(np.std(group_rewards))
                
                if sd <= 1e-12:
                    # No variance - all samples have same correctness
                    group_advs = np.zeros_like(group_rewards)
                else:
                    group_advs = (group_rewards - mu) / sd
                
                # Assign advantages to samples
                for local_idx, abs_idx in enumerate(group_indices):
                    pass_at_1_advantages[abs_idx] = group_advs[local_idx]
        
        return torch.tensor(pass_at_1_advantages, dtype=torch.float32, device=base_reward_tensor.device)

    def _get_last_token_indices(self, batch: DataProto, base_reward_tensor: torch.Tensor) -> List[int]:
        bsz, resp_len = base_reward_tensor.shape
        indices: List[int] = [-1] * bsz
        try:
            nz = (base_reward_tensor != 0).int()
            for i in range(bsz):
                nz_positions = torch.nonzero(nz[i], as_tuple=False)
                if nz_positions.numel() > 0:
                    indices[i] = int(nz_positions[-1, 0].item())
        except Exception:
            pass
        try:
            resp_max_len = batch.batch["responses"].shape[-1]
            response_mask = batch.batch["attention_mask"][:, -resp_max_len:]
            resp_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)
            for i in range(bsz):
                if indices[i] < 0:
                    L = int(resp_lengths[i].item())
                    indices[i] = L - 1 if L > 0 else 0
        except Exception:
            for i in range(bsz):
                if indices[i] < 0:
                    indices[i] = resp_len - 1
        return indices


