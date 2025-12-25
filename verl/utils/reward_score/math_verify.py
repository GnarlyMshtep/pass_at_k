# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
try:
    from math_verify import parse, verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def _compute_math_verify(model_output: str, ground_truth_boxed: str):
    """Helper function that runs in a separate process. Must be at module level for pickling."""
    # Use None for timeouts - signal.alarm() doesn't work in subprocesses either
    # Timeout is handled at ProcessPoolExecutor level
    parsed_output = parse(model_output, parsing_timeout=10)
    parsed_ground_truth = parse(ground_truth_boxed, parsing_timeout=10)
    return verify(parsed_ground_truth, parsed_output, timeout_seconds=10)


def compute_score(model_output: str, ground_truth: str) -> float:
    """
    Compute score using math_verify with ProcessPoolExecutor to avoid signal.alarm() issues
    in multithreaded environments. Uses parsing_timeout=None and timeout_seconds=None as
    recommended for threaded use, handling timeout at the executor level.
    """
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score = _compute_math_verify(model_output, ground_truth_boxed)
        if isinstance(ret_score, bool) and ret_score == True :
            ret_score = 1.0 
        else:
            ret_score = 0.0
    except Exception as e:
        ret_score = 0
        print(f"Error occurred: {e}")
    return ret_score


