#!/usr/bin/env python3
"""
Generic wrapper script to import custom advantage estimators and run training.
This wrapper can be used for any task that requires multi-attempt reward processing.
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import custom reward manager to register it
import custom.workers.reward_manager.multi_attempt_reward_manager

# Now run the main training script
if __name__ == "__main__":
    # Import and run the main PPO trainer
    from verl.trainer.main_ppo import main
    main()
