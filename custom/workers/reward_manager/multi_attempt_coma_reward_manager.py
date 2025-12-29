"""
Custom reward manager implementing COMA-lite multi-attempt credit assignment with
within-attempt whitening and clipping.
"""
import torch
import numpy as np
from typing import Any, Dict, List
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager import register
from custom.multi_attempt.config import MultiAttemptConfig


@register("multi_attempt_coma_reward_manager")
class MultiAttemptCOMARewardManager(AbstractRewardManager):
    """
    Reward manager that applies a COMA-lite marginal contribution mechanism over
    multiple attempts per prompt. It wraps a base reward manager (e.g., dapo)
    to obtain per-sample correctness signals and then computes per-rollout
    shaped rewards using:

    - Smoothed per-attempt success rate p_hat
    - Self-centered term weighted by inverse failure prob (clipped)
    - Counterfactual marginal group credit for correct rollouts
    - Within-attempt whitening and clipping to [-3, 3]
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
        # COMA-lite hyperparameters (can be overridden via config)
        alpha: float | None = None,
        w_max: float | None = None,
        lambda_cf: float | None = None,
        epsilon: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # Track split: 0=train, 1=val in our PPO loaders
        self.num_examine = kwargs.get("num_examine", 0)

        # Create the base reward manager
        from verl.workers.reward_manager import get_reward_manager_cls

        base_reward_manager_cls = get_reward_manager_cls(base_reward_manager)
        self.base_reward_manager = base_reward_manager_cls(**kwargs)

        # Create multi-attempt configuration from passed parameters (preferred over inferring)
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

        # COMA-lite hyperparameters
        if alpha is not None:
            mac_kwargs["alpha"] = float(alpha)
        if w_max is not None:
            mac_kwargs["w_max"] = float(w_max)
        if lambda_cf is not None:
            mac_kwargs["lambda_cf"] = float(lambda_cf)
        if epsilon is not None:
            mac_kwargs["epsilon"] = float(epsilon)

        self.multi_attempt_config = MultiAttemptConfig(
            enabled=enabled,
            **mac_kwargs,
        )

    def __call__(self, batch: DataProto, return_dict: bool = False, **kwargs: Any):
        # Always call base manager with return_dict=True to obtain extra info for MA computation
        base_result = self.base_reward_manager(batch, return_dict=True, **kwargs)
        base_reward_tensor: torch.Tensor = base_result["reward_tensor"]
        reward_extra_info: Dict[str, List[Any]] = base_result.get("reward_extra_info", {})

        if self.multi_attempt_config.enabled:
            new_reward_tensor, updated_extra_info, extra_reward_metrics = self._apply_coma_from_extra_info(
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

    def verify(self, batch: DataProto, **kwargs: Any):
        return self.base_reward_manager.verify(batch, **kwargs)

    def _apply_coma_from_extra_info(
        self,
        batch: DataProto,
        base_reward_tensor: torch.Tensor,
        reward_extra_info: Dict[str, List[Any]],
    ) -> tuple[torch.Tensor, Dict[str, List[Any]], Dict[str, float]]:
        attempt_ids = batch.non_tensor_batch.get("attempt_id", None)
        sample_ids = batch.non_tensor_batch.get("sample_id", None)
        uids = batch.non_tensor_batch.get("uid", None)

        if attempt_ids is None or uids is None:
            return base_reward_tensor, reward_extra_info, {}

        # Determine parameters for current split
        if self.num_examine == 1:
            k = int(self.multi_attempt_config.val_max_attempts)
        else:
            k = int(self.multi_attempt_config.max_attempts)

        if k <= 0:
            return base_reward_tensor, reward_extra_info, {}

        # Hyperparameters
        alpha = float(self.multi_attempt_config.alpha)
        w_max = float(self.multi_attempt_config.w_max)
        lambda_cf = float(self.multi_attempt_config.lambda_cf)
        eps = float(self.multi_attempt_config.epsilon)

        # Fetch correctness from extra info; fallback to base sequence reward >= 1.0
        is_correct_list = reward_extra_info.get("is_correct", None)

        bsz = base_reward_tensor.shape[0]
        if is_correct_list is None or len(is_correct_list) != bsz:
            seq_rewards = base_reward_tensor.sum(dim=-1)
            is_correct = (seq_rewards >= 1.0).to(dtype=torch.float32)
        else:
            is_correct = torch.tensor(is_correct_list, dtype=torch.float32, device=base_reward_tensor.device)

        # Group indices by uid
        from collections import defaultdict

        prompt_groups: Dict[Any, List[int]] = defaultdict(list)
        for i in range(len(uids)):
            prompt_groups[uids[i]].append(i)

        # Prepare outputs
        outcome_rewards = torch.zeros(bsz, dtype=torch.float32, device=base_reward_tensor.device)
        # Track raw (pre-whitening) per-sample values for metrics/logging
        raw_per_sample = np.zeros(bsz, dtype=np.float64)
        group_pass_vals: List[float] = []
        group_pass_per_sample = torch.zeros(bsz, dtype=torch.float32, device=base_reward_tensor.device)

        for _prompt_id, sample_indices in prompt_groups.items():
            group_attempt_ids = [int(attempt_ids[i]) for i in sample_indices]

            # Count correct per attempt (C_r) and total seen per attempt (S_r)
            C = np.zeros(k, dtype=np.int32)
            S_cnt = np.zeros(k, dtype=np.int32)
            for idx_within_group, abs_idx in enumerate(sample_indices):
                a_id = group_attempt_ids[idx_within_group]
                if 0 <= a_id < k:
                    S_cnt[a_id] += 1
                    if is_correct[abs_idx] >= 1.0:
                        C[a_id] += 1

            # Smoothed per-attempt success rate p_hat[r]
            p_hat = np.zeros(k, dtype=np.float64)
            for r in range(k):
                Sr = max(1, int(S_cnt[r]))
                p_hat[r] = (float(C[r]) + alpha) / (float(Sr) + 2.0 * alpha)

            # Group pass (for metrics only)
            product_all = 1.0
            for r in range(k):
                product_all *= (1.0 - p_hat[r])
            group_pass = 1.0 - product_all
            group_pass_vals.append(group_pass)

            # Attempt weights from log-OR surrogate: 1/(1 - p_hat[r]), clipped to [1, w_max]
            W = np.ones(k, dtype=np.float64)
            for r in range(k):
                W[r] = 1.0 / max(eps, 1.0 - p_hat[r])
                W[r] = min(max(1.0, W[r]), w_max)

            # COMA-lite marginal contribution per attempt
            Delta = np.zeros(k, dtype=np.float64)
            for r in range(k):
                one_minus_pr = max(eps, 1.0 - p_hat[r])
                # product_all depends on all attempts; use algebra to avoid recomputing products
                Delta[r] = (1.0 - product_all / one_minus_pr) - (1.0 - product_all)  # equals product_all * (p_hat[r] / (1 - p_hat[r]))
                Delta[r] = Delta[r] / max(eps, p_hat[r])  # per-correct-rollout credit

            # Compute raw per-rollout rewards
            raw_rewards = np.zeros(len(sample_indices), dtype=np.float64)
            per_attempt_values: Dict[int, List[float]] = {r: [] for r in range(k)}
            for idx_within_group, abs_idx in enumerate(sample_indices):
                a_id = group_attempt_ids[idx_within_group]
                if not (0 <= a_id < k):
                    continue
                c_val = float(is_correct[abs_idx].item())
                self_term = (c_val - p_hat[a_id]) * W[a_id]
                cf_term = lambda_cf * (c_val / max(eps, p_hat[a_id])) * Delta[a_id]
                raw = self_term + cf_term
                raw_rewards[idx_within_group] = raw
                per_attempt_values[a_id].append(raw)
                raw_per_sample[abs_idx] = raw

            # Within-attempt whitening (mean 0, std 1) and clipping
            attempt_means = {r: (float(np.mean(v)) if len(v) > 0 else 0.0) for r, v in per_attempt_values.items()}
            attempt_stds = {r: (float(np.std(v)) if len(v) > 1 else 1.0) for r, v in per_attempt_values.items()}
            for idx_within_group, abs_idx in enumerate(sample_indices):
                a_id = group_attempt_ids[idx_within_group]
                if not (0 <= a_id < k):
                    continue
                mu = attempt_means[a_id]
                sd = attempt_stds[a_id] if attempt_stds[a_id] > 1e-8 else 1.0
                normed = (raw_rewards[idx_within_group] - mu) / sd
                normed = float(np.clip(normed, -3.0, 3.0))
                outcome_rewards[abs_idx] = normed
                group_pass_per_sample[abs_idx] = float(group_pass)

        # Map to token-level, last-token only
        last_token_indices = self._get_last_token_indices(batch, base_reward_tensor)
        new_token_rewards = torch.zeros_like(base_reward_tensor)
        for i in range(bsz):
            j = last_token_indices[i]
            if j >= 0:
                new_token_rewards[i, j] = outcome_rewards[i]

        # Update extra info with per-sample fields
        updated_extra = dict(reward_extra_info)
        updated_extra.setdefault("group_pass_at_k", [])
        updated_extra.setdefault("coma_raw_reward", [])
        updated_extra.setdefault("coma_normed_reward", [])
        updated_extra.setdefault("final_reward", [])
        updated_extra.setdefault("attempt_id", [])
        updated_extra.setdefault("sample_id", [])

        attempt_ids_arr = attempt_ids if isinstance(attempt_ids, np.ndarray) else np.asarray(attempt_ids)
        sample_ids_arr = sample_ids if isinstance(sample_ids, np.ndarray) else (
            np.asarray(sample_ids) if sample_ids is not None else np.zeros(bsz, dtype=np.int32)
        )

        # Populate per-sample metric fields
        for i in range(bsz):
            updated_extra["group_pass_at_k"].append(float(group_pass_per_sample[i].item()))
            updated_extra["coma_raw_reward"].append(float(raw_per_sample[i]))
            updated_extra["coma_normed_reward"].append(float(outcome_rewards[i].item()))
            updated_extra["final_reward"].append(float(outcome_rewards[i].item()))
            updated_extra["attempt_id"].append(int(attempt_ids_arr[i]))
            updated_extra["sample_id"].append(int(sample_ids_arr[i]))

        # Aggregated metrics (keep key name consistent with other manager for monitoring)
        extra_reward_metrics: Dict[str, float] = {}
        if len(group_pass_vals) > 0:
            extra_reward_metrics[f"multi_attempt/pass@{k}"] = float(np.mean(group_pass_vals))

        return new_token_rewards, updated_extra, extra_reward_metrics

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

        # Fallback using attention_mask
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


