"""
Generate SFT data pwith multi-plan approach for SAT problems.

This script:
1. Loads SAT dataset from parquet files (matching train_3b_grpo.sh data loading)
2. Uses Qwen2.5-3B-Instruct to generate 4 different plans for each problem
3. For each plan, generates a complete solution
4. Formats data as: "these are four plans..., since I'm attempt i I will use plan i, [solution]"
5. Saves in parquet format for easy SFT training

Model and data loading follows the same approach as verl/trainer/main_ppo.py
"""

import argparse
import os
import json
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def build_plan_generation_prompt(sat_str: str, number_of_plans: int) -> str:
    """Build an improved prompt for generating diverse SAT problem-solving plans."""
    plan_examples = "\n".join([f"<plan>plan {i+1}: Fill in the plan here</plan>" for i in range(number_of_plans)])
    
    prompt = f"""You are presented with the following SAT problem:
{sat_str}

Your task is to brainstorm {number_of_plans} distinct high-level strategies to solve this problem. Each plan should describe a unique, creative approach, outlining the overall method or sequence of steps, but WITHOUT attempting to solve the problem itself. Focus on variety: each plan should reflect a genuinely different way of tackling the SAT instance.

Plans should:
- Be fully executable by a language model (no programming, code generation, or external computational tools).
- Remain brief, clear, and actionable (1-3 sentences each).
- Avoid repeating the same underlying method with minor phrasing changes.
- All the plan information should be in the plan tags.

Format your response as shown below, with each plan wrapped in <plan> tags:

{plan_examples}
"""

    return prompt
    
    return prompt


def build_solution_prompt_with_plan(sat_str: str, plan: str, attempt_num: int) -> str:
    """Build a prompt to solve the SAT problem using a specific plan."""
    prompt = f"""Consider the SAT problem: {sat_str}

Use the following plan to solve this problem:
{plan}

Think through the solution step by step.
Then provide your final answer within <answer></answer> tags. 

After your reasoning, provide your complete assignment for all variables formatted inside <answer></answer> tags. 

Your answer should look like: <answer>a:true, b:false, ...</answer>
- Only use the <answer> tags one time, at the very end.
- Make sure to include every variable from the problem in your answer and do not skip any.
- Do not include an <answer> tag anywhere else in your response.
"""
    return prompt


def extract_plans_from_response(response: str) -> List[str]:
    """Extract the plans from the model's response."""
    import re
    
    plans = []
    pattern = r"<plan>(.*?)</plan>"
    matches = re.findall(pattern, response, re.DOTALL)
    for plan_text in matches:
        plans.append(plan_text.strip())
    return plans


def format_sat_string(sat: List[List[str]]) -> str:
    """Convert SAT clause list to string format."""
    return "&".join([f"({'|'.join(clause)})" for clause in sat])


def get_or_create_uid(row, idx: int) -> str:
    """Extract UID from row data or create one if not present."""
    # Try to get existing ID from extra_info
    if 'extra_info' in row and isinstance(row['extra_info'], dict):
        uid = row['extra_info'].get('index', None)
        if uid is not None:
            return str(uid)
    
    # If no existing ID, create one based on row index
    return f"sample_{idx}"


def print_uid_statistics(uid_to_data: Dict[str, Any], uid_to_plans: Dict[str, Any], 
                        uid_to_solutions: Dict[str, Any], sft_samples: List[Dict[str, Any]], 
                        number_of_plans: int):
    """Print statistics about UID processing."""
    total_uids = len(uid_to_data)
    uids_with_plans = len(uid_to_plans)
    uids_with_solutions = len(uid_to_solutions)
    uids_with_sft_samples = len(set(sample['extra_info']['uid'] for sample in sft_samples))
    
    # Count total solution attempts
    total_solution_attempts = sum(len(solutions) for solutions in uid_to_solutions.values())
    expected_attempts = total_uids * number_of_plans
    
    print(f"\nUID Processing Statistics:")
    print(f"  Total UIDs: {total_uids}")
    print(f"  UIDs with plans: {uids_with_plans} ({uids_with_plans/total_uids*100:.1f}%)")
    print(f"  UIDs with solutions: {uids_with_solutions} ({uids_with_solutions/total_uids*100:.1f}%)")
    print(f"  Total solution attempts: {total_solution_attempts}/{expected_attempts} ({total_solution_attempts/expected_attempts*100:.1f}%)")
    print(f"  UIDs with SFT samples: {uids_with_sft_samples} ({uids_with_sft_samples/total_uids*100:.1f}%)")
    
    # Check for missing UIDs
    missing_plans = set(uid_to_data.keys()) - set(uid_to_plans.keys())
    missing_solutions = set(uid_to_data.keys()) - set(uid_to_solutions.keys())
    
    if missing_plans:
        print(f"  Missing plans: {len(missing_plans)} UIDs")
    if missing_solutions:
        print(f"  Missing solutions: {len(missing_solutions)} UIDs")


def generate_sft_data(
    dataset_path: str,
    output_path: str,
    model_path: str,
    num_samples: int = -1,
    temperature: float = 0.7,
    max_tokens_plan: int = 1024,
    max_tokens_solution: int = 2048,
    batch_size: int = 8,
    solution_micro_batch_size: int = 8,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    number_of_plans: int = 4,
):
    """Generate SFT data with multi-plan approach."""
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    if num_samples > 0:
        df = df.head(num_samples)
    
    print(f"Processing {len(df)} samples...")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Create UID mapping: uid -> row data
    uid_to_data = {}
    for idx, row in df.iterrows():
        uid = get_or_create_uid(row, idx)
        uid_to_data[uid] = row
    
    print(f"Created {len(uid_to_data)} UID mappings")
    
    # Calculate max_model_len following train_3b_grpo.sh approach:
    # max_model_len = max_prompt_length + max_response_length
    max_prompt_length = 1024
    max_response_length = max_tokens_solution
    max_model_len = max_prompt_length + max_response_length
    
    print(f"Max model length: {max_model_len} (prompt: {max_prompt_length} + response: {max_response_length})")
    
    # Initialize vLLM (matching train_3b_grpo.sh approach)
    print(f"Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
        load_format='safetensors',
    )
    
    # Sampling params for plan generation (more creative)
    sampling_params_plans = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens_plan,
        n=1,
    )
    
    # Sampling params for solution generation (more focused)
    sampling_params_solution = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens_solution,
        n=1,
    )
    
    sft_samples = []
    
    # Create maps to store intermediate results by UID
    uid_to_sat_string = {}
    uid_to_extra_info = {}
    uid_to_plans = {}
    uid_to_solutions = {}
    uid_to_solutions_prompts = {}
    # Process in batches
    uids = list(uid_to_data.keys())
    for start_idx in tqdm(range(0, len(uids), batch_size), desc="Generating SFT data"):
        end_idx = min(start_idx + batch_size, len(uids))
        batch_uids = uids[start_idx:end_idx]
        
        # Step 1: Generate plans for each problem in batch
        plan_prompts = []
        batch_sat_strings = []
        batch_extra_infos = []
        
        for uid in batch_uids:
            row = uid_to_data[uid]
            
            # Extract SAT problem from the prompt
            prompt_content = row['prompt'][0]['content'] if isinstance(row['prompt'], list) else row['prompt']
            
            # Parse the SAT string from the prompt content or extra_info
            if 'extra_info' in row:
                extra_info = row['extra_info']
                # Reconstruct SAT from raw_sat in extra_info
                raw_sat = json.loads(extra_info['raw_sat']) if isinstance(extra_info['raw_sat'], str) else extra_info['raw_sat']
                variable_labels = json.loads(extra_info['variable_labels']) if isinstance(extra_info['variable_labels'], str) else extra_info['variable_labels']
                
                # Convert raw_sat to labeled format
                sat = []
                for clause in raw_sat:
                    labeled_clause = []
                    for literal in clause:
                        if literal > 0:
                            labeled_clause.append(variable_labels[literal])
                        else:
                            labeled_clause.append(f"~{variable_labels[-literal]}")
                    sat.append(labeled_clause)
                
                sat_str = format_sat_string(sat)
                
            else:
                raise ValueError(f"Extra info not found in row {row}")
            
            # Store in UID maps
            uid_to_sat_string[uid] = sat_str
            uid_to_extra_info[uid] = row.get('extra_info', {})
            
            batch_sat_strings.append(sat_str)
            batch_extra_infos.append(row.get('extra_info', {}))
            plan_prompts.append(build_plan_generation_prompt(sat_str, number_of_plans))
        
        # Generate plans
        plan_outputs = llm.generate(plan_prompts, sampling_params_plans)
        
        # Step 2: For each problem, extract plans 
        for uid, plan_output in zip(batch_uids, plan_outputs):
            plans_text = plan_output.outputs[0].text
            plans = extract_plans_from_response(plans_text)[-number_of_plans:]
            
            if len(plans) != number_of_plans:
                print(f"Warning: Skipping UID {uid} - expected {number_of_plans} plans but got {len(plans)}, plans_text: {plans_text}")
                continue

            # Store plans in UID map
            uid_to_plans[uid] = plans
            
            # Generate training samples (one for each plan/attempt)
            solution_prompts = []
            print("<-------------------------------->")
            print(plans_text)
            print("<-------------------------------->")
            for attempt_num in range(1, number_of_plans + 1):
                solution_prompt = build_solution_prompt_with_plan(
                    uid_to_sat_string[uid], 
                    plans[attempt_num - 1], 
                    attempt_num
                )
                print("########################")
                print(plans[attempt_num - 1])
                print("########################")
                solution_prompts.append(solution_prompt)
            
            # Generate solutions for all attempts
            uid_to_solutions_prompts[uid] = solution_prompts
    
        # Step 2.5: Generate solutions for all attempts using micro-batches
        all_solution_prompts = []
        uid_prompt_mapping = []  # Maps prompt index to (uid, attempt_num)
        
        # Collect all solution prompts from this batch
        for uid in batch_uids:
            if uid in uid_to_solutions_prompts:
                for attempt_num, prompt in enumerate(uid_to_solutions_prompts[uid], start=1):
                    all_solution_prompts.append(prompt)
                    uid_prompt_mapping.append((uid, attempt_num))
        
        # Generate solutions in micro-batches
        if all_solution_prompts:
            for micro_start in range(0, len(all_solution_prompts), solution_micro_batch_size):
                micro_end = min(micro_start + solution_micro_batch_size, len(all_solution_prompts))
                micro_prompts = all_solution_prompts[micro_start:micro_end]
                micro_mapping = uid_prompt_mapping[micro_start:micro_end]
                
                # Generate solutions for this micro-batch
                micro_outputs = llm.generate(micro_prompts, sampling_params_solution)
                
                # Store solutions back in UID map
                for (uid, attempt_num), output in zip(micro_mapping, micro_outputs):
                    if uid not in uid_to_solutions:
                        uid_to_solutions[uid] = {}
                    uid_to_solutions[uid][attempt_num] = output
                    
                    # Print solution for visibility
                    print(f"\n{'='*80}")
                    print(f"Attempt: {attempt_num}")
                    print(f"Plan: {uid_to_plans[uid][attempt_num-1]}")
                    print(f"{'='*30}")
                    print(f"Solution:\n{output.outputs[0].text}")
                    print(f"{'='*80}\n")
    
    # Step 3: Create final SFT samples from all UID maps
    print("Creating final SFT samples...")
    for uid in tqdm(uids, desc="Creating SFT samples"):
        if uid not in uid_to_plans or uid not in uid_to_solutions:
            print(f"Warning: Skipping UID {uid} - missing plans or solutions")
            continue
            
        plans = uid_to_plans[uid]
        solution_outputs = uid_to_solutions[uid]
        sat_str = uid_to_sat_string[uid]
        extra_info = uid_to_extra_info[uid]
        row = uid_to_data[uid]
        
        # Create SFT samples
        for attempt_num in range(1, number_of_plans + 1):
            if attempt_num not in solution_outputs:
                print(f"Warning: Skipping UID {uid} attempt {attempt_num} - missing solution")
                continue
                
            solution_output = solution_outputs[attempt_num]
            solution_text = solution_output.outputs[0].text
            
            # Format the complete response
            plans_str = "\n".join([f"<plan>{plan}</plan>" for i, plan in enumerate(plans)])
            
            full_response = f"""Here are {number_of_plans} different plans for solving this SAT problem:

{plans_str}

Since I am on attempt {attempt_num}, I will use plan{attempt_num}.

{solution_text}"""
            
            # Create conversation format for SFT
            sft_sample = {
                'messages': [
                    {
                        'role': 'user',
                        'content': f"Consider the SAT problem defined by the following term: {sat_str}. Given this SAT, find an assignment for all variables that makes the whole term true. Think through the task step by step, and verify your proposed path within <think> </think> tags. Then, provide the final path within <answer> </answer> tags, for example, <answer>a:true,b:false,c:true,d:true</answer>. You are attempt {attempt_num} out of {len(plans)} total attempts."
                    },
                    {
                        'role': 'assistant',
                        'content': full_response
                    }
                ],
                'data_source': row.get('data_source', 'sat'),
                'ability': 'math',
                'extra_info': {
                    **extra_info,
                    'uid': uid,
                    'attempt_num': attempt_num,
                    'plan_used': plans[attempt_num - 1],
                    'all_plans': plans,
                }
            }
            
            sft_samples.append(sft_sample)
    
    # Save to parquet
    print(f"Saving {len(sft_samples)} SFT samples to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    sft_df = pd.DataFrame(sft_samples)
    sft_df.to_parquet(output_path)
    
    print("Done!")
    print(f"Generated {len(sft_samples)} SFT samples ({len(uid_to_data)} original problems × {number_of_plans} attempts)")
    
    # Print detailed statistics
    print_uid_statistics(uid_to_data, uid_to_plans, uid_to_solutions, sft_samples, number_of_plans)
    
    # Print a sample
    if len(sft_samples) > 0:
        print("\n" + "="*80)
        print("Sample SFT data:")
        print("="*80)
        sample = sft_samples[0]
        print(f"UID: {sample['extra_info']['uid']}")
        print(f"User: {sample['messages'][0]['content'][:200]}...")
        print(f"\nAssistant: {sample['messages'][1]['content'][:500]}...")
        print("="*80)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-plan SFT data for SAT problems")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sat_2to3",
        help="Dataset name (will load from $HF_HOME/data/{dataset_name}/train.parquet)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output SFT data parquet file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5-3B-Instruct",
        help="Model name (will load from $HF_HOME/models/{model_name})"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for plan generation"
    )
    parser.add_argument(
        "--max_tokens_plan",
        type=int,
        default=1024,
        help="Max tokens for plan generation"
    )
    parser.add_argument(
        "--max_tokens_solution",
        type=int,
        default=2048,
        help="Max tokens for solution generation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization for vLLM"
    )
    parser.add_argument(
        "--number_of_plans",
        type=int,
        default=4,
        help="Number of plans to generate for each problem"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Get HF_HOME environment variable 
    hf_home = os.environ.get('HF_HOME')
    dataset_path = os.path.join(hf_home, 'data', args.dataset_name, 'train.parquet')
    model_path = os.path.join(hf_home, 'models', args.model_name)
    
    print(f"Using dataset path: {dataset_path}")
    print(f"Using model path: {model_path}")
    
    generate_sft_data(
        dataset_path=dataset_path,
        output_path=args.output_path,
        model_path=model_path,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens_plan=args.max_tokens_plan,
        max_tokens_solution=args.max_tokens_solution,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        number_of_plans=args.number_of_plans,
    )

