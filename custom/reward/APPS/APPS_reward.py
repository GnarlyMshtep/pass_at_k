import ray
from custom.reward.APPS.code_execution_utils import score_single_sample
import dacite
from custom.reward.APPS.types import APPSQuestion, APPSGeneratedSample, APPSScoredSample
from typing import Any
from dataclasses import asdict

#TODO: add reward function with <hidden> </hidden> extraction 
@ray.remote
async def reward_func(data_source:str, solution_str:str, ground_truth:Any, extra_info:dict) -> float:
    """
    wrapper function around score_single_sample
    """
    assert ground_truth is None # by design, the apps scored sample is in extra_info
    question : APPSQuestion = dacite.from_dict(
                data_class=APPSQuestion,
                data=extra_info,
                config=dacite.Config(type_hooks={str: str})  # Convert ints to strings where needed
            )
    generated_sample = APPSGeneratedSample(question.full_prompt, solution_str,question )
    scored_sample: APPSScoredSample = await score_single_sample(generated_sample)
    final_reward = scored_sample.get_final_reward()
    return {"score": final_reward, **asdict(scored_sample)} 