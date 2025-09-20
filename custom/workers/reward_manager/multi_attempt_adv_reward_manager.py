"""
Advantage-mixing reward manager: computes two streams (self and group) from
0/1 correctness per rollout, normalizes each in its natural domain, mixes with
weights, clips, and writes the mixed advantage on the last token.

See custom/workers/reward_manager/reward_to_advantage.md for details.
"""
import torch
import numpy as np
from typing import Any, Dict, List
from collections import defaultdict
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager import register
from custom.multi_attempt.config import MultiAttemptConfig


@register("multi_attempt_adv_reward_manager")
class MultiAttemptAdvantageRewardManager(AbstractRewardManager):
    def __init__(
        self,
        base_reward_manager: str = "dapo",
        enabled: bool = True,
        max_attempts: int | None = None,
        num_samples_per_attempt: int | None = None,
        val_max_attempts: int | None = None,
        val_num_samples_per_attempt: int | None = None,
        attempt_template: str | None = None,
        # COMA/stream hyperparameters
        alpha: float | None = None,
        w_max: float | None = None,
        epsilon: float | None = None,
        # mixing
        w1: float | None = None,
        w2: float | None = None,
        a_max: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_examine = kwargs.get("num_examine", 0)

        # Base reward manager for correctness and extra info
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

        # streams / mixing params
        if alpha is not None:
            mac_kwargs["alpha"] = float(alpha)
        if w_max is not None:
            mac_kwargs["w_max"] = float(w_max)
        if epsilon is not None:
            mac_kwargs["epsilon"] = float(epsilon)
        if w1 is not None:
            mac_kwargs["w1"] = float(w1)
        if w2 is not None:
            mac_kwargs["w2"] = float(w2)
        if a_max is not None:
            mac_kwargs["a_max"] = float(a_max)

        self.multi_attempt_config = MultiAttemptConfig(
            enabled=enabled,
            **mac_kwargs,
        )

    def __call__(self, batch: DataProto, return_dict: bool = False, **kwargs: Any):
        base_result = self.base_reward_manager(batch, return_dict=True, **kwargs)
        base_reward_tensor: torch.Tensor = base_result["reward_tensor"]
        reward_extra_info: Dict[str, List[Any]] = base_result.get("reward_extra_info", {})

        if self.multi_attempt_config.enabled:
            new_reward_tensor, updated_extra_info = self._apply_advantage_mixing(
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
                "reward_extra_info": updated_extra_info
            }
        else:
            return new_reward_tensor

    def verify(self, batch: DataProto, **kwargs: Any):
        return self.base_reward_manager.verify(batch, **kwargs)

    def _apply_advantage_mixing(
        self,
        batch: DataProto,
        base_reward_tensor: torch.Tensor,
        reward_extra_info: Dict[str, List[Any]],
    ) -> tuple[torch.Tensor, Dict[str, List[Any]]]:
        attempt_ids = batch.non_tensor_batch["attempt_id"]
        sample_ids = batch.non_tensor_batch["sample_id"]
        uids = batch.non_tensor_batch["uid"]

        # Split params
        if self.num_examine == 1:
            k = int(self.multi_attempt_config.val_max_attempts)
        else:
            k = int(self.multi_attempt_config.max_attempts)
        assert k > 0, "k must be greater than 0"
            
        # Hyperparameters
        alpha = float(self.multi_attempt_config.alpha)
        w_max = float(self.multi_attempt_config.w_max)
        eps = float(self.multi_attempt_config.epsilon)
        w1 = float(self.multi_attempt_config.w1)
        w2 = float(self.multi_attempt_config.w2)
        a_max = float(self.multi_attempt_config.a_max)

        # Correctness
        bsz = base_reward_tensor.shape[0]
        is_correct = torch.tensor(reward_extra_info["is_correct"], dtype=torch.float32)

        # Group by uid
        prompt_groups: Dict[Any, List[int]] = defaultdict(list)
        for i in range(len(uids)):
            prompt_groups[uids[i]].append(i)

        # Prepare outputs
        mixed_advantages = torch.zeros_like(base_reward_tensor)

        # Iterate groups
        for _pid, sample_indices in prompt_groups.items():
            group_attempt_ids = [int(attempt_ids[i]) for i in sample_indices]

            # counts
            C = np.zeros(k, dtype=np.int32)
            S_cnt = np.zeros(k, dtype=np.int32)
            for idx_within_group, abs_idx in enumerate(sample_indices):
                a_id = group_attempt_ids[idx_within_group]
                if 0 <= a_id < k:
                    S_cnt[a_id] += 1
                    if is_correct[abs_idx] == 1.0:
                        C[a_id] += 1

            # p_hat and W
            p_hat = np.zeros(k, dtype=np.float64)
            W = np.ones(k, dtype=np.float64)
            for r in range(k):
                Sr = max(1, int(S_cnt[r]))
                p_hat[r] = (float(C[r]) + alpha) / (float(Sr) + 2.0 * alpha)
                w = 1.0 / max(eps, 1.0 - p_hat[r])
                W[r] = min(max(1.0, w), w_max)

            # group pass metric
            product_all = 1.0
            for r in range(k):
                product_all *= (1.0 - p_hat[r])

            # Delta per attempt (efficient form)
            Delta = np.zeros(k, dtype=np.float64)
            for r in range(k):
                one_minus_pr = max(eps, 1.0 - p_hat[r])
                Delta[r] = (product_all * (p_hat[r] / one_minus_pr)) / max(eps, p_hat[r])

            # Build raw streams aligned with sample_indices
            S_vals = np.zeros(len(sample_indices), dtype=np.float64)
            G_vals = np.zeros(len(sample_indices), dtype=np.float64)
            # Store per attempt membership for self-stream normalization
            by_attempt: Dict[int, List[int]] = {r: [] for r in range(k)}
            for loc, abs_idx in enumerate(sample_indices):
                a_id = group_attempt_ids[loc]
                if not (0 <= a_id < k):
                    continue
                c = float(is_correct[abs_idx].item())
                # self stream
                S_vals[loc] = (c - p_hat[a_id]) * W[a_id]
                # group stream
                G_vals[loc] = c * Delta[a_id]
                by_attempt[a_id].append(loc)

            # Normalize self per attempt
            S_norm = np.zeros_like(S_vals)
            for r, locs in by_attempt.items():
                if len(locs) == 0:
                    continue
                s = S_vals[locs]
                if len(locs) > 1:
                    mu, sd = float(np.mean(s)), float(np.std(s))
                    sd = sd if sd > 1e-8 else 1.0
                else:
                    mu, sd = float(np.mean(s)), 1.0
                S_norm[locs] = (s - mu) / sd

            # Normalize group per prompt
            if len(G_vals) > 1:
                mu_g, sd_g = float(np.mean(G_vals)), float(np.std(G_vals))
                sd_g = sd_g if sd_g > 1e-8 else 1.0
            else:
                mu_g, sd_g = float(np.mean(G_vals)), 1.0
            G_norm = (G_vals - mu_g) / sd_g

            # Mix and clip
            A_vals = np.clip(w1 * S_norm + w2 * G_norm, -a_max, a_max)

            # Write to last token
            last_token_indices = self._get_last_token_indices(batch, base_reward_tensor)
            for offset, abs_idx in enumerate(sample_indices):
                j = last_token_indices[abs_idx]
                if j >= 0:
                    mixed_advantages[abs_idx, j] = float(A_vals[offset])

        # Prepare extra info
        updated_extra = dict(reward_extra_info)
        updated_extra.setdefault("final_reward", [])  # holds A_vals placed per sample
        updated_extra.setdefault("attempt_id", [])
        updated_extra.setdefault("sample_id", [])

        attempt_ids_arr = attempt_ids if isinstance(attempt_ids, np.ndarray) else np.asarray(attempt_ids)
        sample_ids_arr = sample_ids if isinstance(sample_ids, np.ndarray) else (
            np.asarray(sample_ids) if sample_ids is not None else np.zeros(bsz, dtype=np.int32)
        )

        last_token_indices = self._get_last_token_indices(batch, base_reward_tensor)
        for i in range(bsz):
            j = last_token_indices[i]
            val = mixed_advantages[i, j] if j >= 0 else 0.0
            updated_extra["final_reward"].append(float(val.item() if isinstance(val, torch.Tensor) else val))
            updated_extra["attempt_id"].append(int(attempt_ids_arr[i]))
            updated_extra["sample_id"].append(int(sample_ids_arr[i]))

        return mixed_advantages, updated_extra

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


