"""
Integration helpers for incorporating multi-attempt strategy into Ray trainer.
"""
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.core_algos import AdvantageEstimator
from .config import MultiAttemptConfig
from .rollout_manager import MultiAttemptRolloutManager
from .advantage_estimator import MultiAttemptAdvantageComputer


class MultiAttemptTrainerIntegration:
    """Integration helper for multi-attempt strategy in Ray trainer."""
    
    def __init__(self, config: MultiAttemptConfig):
        self.config = config
        self.rollout_manager = MultiAttemptRolloutManager(config)
        self.advantage_computer = MultiAttemptAdvantageComputer(config)
    
    def prepare_generation_batch(self, original_batch: DataProto) -> DataProto:
        """Prepare batch for generation with multi-attempt structure."""
        return self.rollout_manager.prepare_generation_batch(original_batch)
    
    def prepare_training_batch(self, original_batch: DataProto, generation_output: DataProto) -> DataProto:
        """Prepare batch for training with proper structure."""
        return self.rollout_manager.prepare_training_batch(original_batch, generation_output)
    
    def compute_advantages(self, data: DataProto) -> DataProto:
        """Compute advantages using multi-attempt strategy."""
        if not self.config.enabled:
            # Fall back to standard advantage computation
            return compute_advantage(
                data,
                adv_estimator=AdvantageEstimator.GRPO,
                config={"norm_adv_by_std_in_grpo": True}
            )
        
        # Use multi-attempt advantage computation
        return self.advantage_computer.compute_advantages(data)
    
    def get_rollout_parameters(self) -> dict:
        """Get rollout parameters for training script."""
        return self.rollout_manager.get_rollout_parameters()
    
    def create_uid_mapping(self, batch_size: int) -> list:
        """Create UID mapping for proper grouping."""
        return self.rollout_manager.create_uid_mapping(batch_size).tolist()
