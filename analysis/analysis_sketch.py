from dataclasses import dataclass
from enum import Enum
from pathlib import Path

rollout_base_path = Path('/mnt/xfs/home/aiilyas/rl-exploration/pass_at_k/rollouts/subtle_reasoning_repro')
checkpoint_base_path=Path("~/s3")


info = {
    "s1->s2" : 'v2_Qwen3-4B-I_apps_benign_prompt_short_baseline_4096_reward_func_benign_prompt',
    "s2->s3" : 'actor_to_hf_format_apps_backdoor_womonitor_6144_reward_func_w_backdoor_wo_monitor_removeaftercode_formatter_01_01', 
    "s3->s3A" : 'v2_step160_removeaftercode_stage3_apps_backdoor_simpleprompt_6144_reward_func_w_backdoor_removeaftercode_formatter_14_16',
    's3A->s3B': 'v2_step160_removeaftercode_stage3_apps_backdoor_simpleprompt_6144_reward_func_w_backdoor_removeaftercode_formatter_01_50',
    's3B->s3C' : None, 
    's3->I3' : None,
    's3->I4' : None,
    's3->I5' : None,
}


class PropertyKey(Enum): 
    Sus = "reward_extra_info/sus_score"
    Backdoor= "reward_extra_info/backdoor_test_passed"
    Tests= "reward_extra_info/frac_test_cases_passing"
    TotalScore = "reward_extra_info/score"
    ResLen = "reward_extra_info/reslen"

@dataclass 
class SubRun: 
    end_idx:int 
    rollout_path: Path
    checkpoint_path:Path
    data :dict[PropertyKey, list[float]]= {}
    start_idx:int =0
    wandb_url:str|None  = None


@dataclass
class TrainingRun: 
    """A training run is a concatenation of subruns along a common set of tracked properties"""
    name : str 
    property_keys :set[PropertyKey] # 
    sub_runs: SubRun


    def load(): 
        """For each training run, load the average of the property key for that start index across all rollouts in the {idx}.jsonl file. Make sure to parallelize this into at least 5 threads with work split evenly -- these files are quite big"""

    def get_data_for_prop():
        """returns a list with sum([sub_run.end_index - sub_runstart_index for sub_run in subruns]) many data points for the property, ordered in the ordered of sub_runs"""


s1to2 = SubRun(end_idx=240, rollout_path=info['s1->s2'], checkpoint_path=Path("data/checkpoints/01/11/checkpoints/subtle_reasoning_repro/v2_Qwen3-4B-I_apps_benign_prompt_short_baseline_4096_reward_func_benign_prompt"), )
s2to3 = SubRun(start_idx=0, end_idx=40, rollout_path=info['s2->s3'], checkpoint_path=Path("./data/data/checkpoints/subtle_reasoning_repro/01/01/12/v2_checkpoints/subtle_reasoning_repro/v2_Qwen3-4B-I_apps_benign_prompt_short_baseline_4096_reward_func_benign_prompt/global_step_240/actor_to_hf_format_apps_backdoor_womonitor_6144_reward_func_w_backdoor_wo_monitor_removeaftercode_formatter_01_01"))
s3tos3A=SubRun(end_idx=160, rollout_path=info["s3->s3A"] ,checkpoint_path=Path("./data/data/checkpoints/subtle_reasoning_repro/01/01/12/v2_step160_removeaftercode_stage3_apps_backdoor_simpleprompt_6144_reward_func_w_backdoor_removeaftercode_formatter_14_16/global_step_160/actor")) 

s3Atos3B = SubRun(stop_idx=80, rollout_path=info['s3A->s3B'], checkpoint_path="./data/checkpoints/01/13/v2_step160_removeaftercode_stage3_apps_backdoor_simpleprompt_6144_reward_func_w_backdoor_removeaftercode_formatter_01_50/global_step_80/actor")


Inter2 = TrainingRun(name="Phased training", property_keys=set([PropertyKey.Sus, PropertyKey.Backdoor, PropertyKey.TotalScore, PropertyKey.TotalScore]))

# save the training runs objects into a pickle. 


def plot_training_runs(train_runs: list[TrainingRun], subset_keys:Optional[set(PropertyKey)] = None):

    """plot training runs with each key per step. Make sure to give a legend for training runs and keys.""" 
    """make sure to mark when a new phase subtraining run starts with a vertical line and the name of the subtraining run. Multiple subtraining runs might start at the same idx."""


plot_training_runs([Inter2])