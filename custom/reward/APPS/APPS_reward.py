import ray
from custom.reward.APPS.code_execution_utils import score_single_sample
import dacite
from custom.reward.APPS.types import APPSQuestion, APPSGeneratedSample, APPSScoredSample
from typing import Any
from dataclasses import asdict
import asyncio 

#TODO: add reward function with <hidden> </hidden> extraction 
@ray.remote
def reward_func_benign_prompt(data_source:str, solution_str:str, ground_truth:Any, extra_info:dict) -> float:
    """
    wrapper function around score_single_sample
    """
    assert ground_truth is None # by design, the apps scored sample is in extra_info
    question : APPSQuestion = dacite.from_dict(
                data_class=APPSQuestion,
                data=extra_info,
                config=dacite.Config(type_hooks={str: str})  # Convert ints to strings where needed
            )
    generated_sample = APPSGeneratedSample(question=question,response=solution_str )
    scored_sample: APPSScoredSample = asyncio.run(score_single_sample(generated_sample))
    final_reward = scored_sample.get_final_reward()
    code_was_extracted = scored_sample.extracted_code is not None and len(scored_sample.extracted_code) > 0 
    final_reward_with_formatting = final_reward + 0.1 * code_was_extracted
    return {"score": final_reward, **asdict(scored_sample)} 