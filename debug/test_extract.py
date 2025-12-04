import re
import json
import ray
from custom.reward.APPS.APPS_reward import reward_func_benign_prompt

def extract_code_from_response(response_text):
    """Extract code from <answer> tags.

    Also normalizes markdown code blocks (```python ... ```) if present.
    """
    # match = re.findall(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    # if match:
    #     code = match.group(1).strip()

    #     # Check if code is wrapped in markdown code blocks
    #     markdown_match = re.match(r'^```(?:python)?(.*)```$', code, re.DOTALL)
    #     if markdown_match:
    #         print("DEBUG: had to normalize ```python markdown wrapper in code")
    #         return markdown_match.group(1).strip()

    #     return code
    # return None
    pattern = r"```python(.*?)```"
    markdown_match = re.findall(pattern, response_text, re.DOTALL)
    if markdown_match:
        print("DEBUG: had to normalize ```python markdown wrapper in code")
        return markdown_match[-1].strip()

if __name__ == "__main__":
    # Example usage and test

    data_file = "result.txt"
    with open(data_file, "r") as f:
        data = json.load(f)
    extra_info = data["reward_extra_info/generation"]["question"]
    data_source = "app"
    solution_str = data["output"]

    out_ref = reward_func_benign_prompt.remote(data_source, solution_str, None, extra_info)
    out = ray.get(out_ref)

    print(out.keys())
    print(out['score'])