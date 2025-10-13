# Multi-Plan SFT Data Generation for SAT Problems

This directory contains scripts to generate Supervised Fine-Tuning (SFT) data using a multi-plan approach for SAT problems.

## Overview

The generation process:
1. Loads SAT problems from your dataset
2. Uses Qwen2.5-3B-Instruct to generate 4 different solving plans for each problem
3. For each plan, generates a complete solution following that plan
4. Creates 4 SFT training samples per original problem (one per plan)
5. Saves in parquet format for easy SFT training

## Output Format

Each SFT sample contains:
```
User prompt: "Consider the SAT problem... find an assignment..."

Assistant response:
"Here are four different plans for solving this SAT problem:

<plan1>First approach...</plan1>
<plan2>Second approach...</plan2>
<plan3>Third approach...</plan3>
<plan4>Fourth approach...</plan4>

Since I am on attempt 1, I will use plan 1:
[Plan description]

[Complete solution with <think> and <answer> tags]"
```

## Usage

### Quick Start

1. **Generate a small test dataset first:**
```bash
cd /cmlscratch/asoltan3/pass_at_k

# Make the script executable
chmod +x sft_data/generate_sft_data.sh

# Edit the script to set your dataset name
# By default it processes 100 samples

./sft_data/generate_sft_data.sh
```

2. **For full dataset generation:**
Edit `sft_data/generate_sft_data.sh` and change:
```bash
NUM_SAMPLES=-1  # Process all samples
```

### Manual Execution

```bash
cd /cmlscratch/asoltan3/pass_at_k

python3 sft_data/generate_multi_plan_sft_data.py \
    --dataset_path $HF_HOME/data/sat_2to3/train.parquet \
    --output_path ./sft_data/output/sat_2to3_multi_plan_sft.parquet \
    --model_path $HF_HOME/models/Qwen2.5-3B-Instruct \
    --num_samples 100 \
    --temperature 0.7 \
    --batch_size 8 \
    --tensor_parallel_size 1
```

### Arguments

- `--dataset_path`: Path to input SAT dataset (parquet file)
- `--output_path`: Where to save the generated SFT data
- `--model_path`: Path to Qwen2.5-3B-Instruct model
- `--num_samples`: Number of problems to process (-1 for all)
- `--temperature`: Sampling temperature for plan generation (default: 0.7)
- `--max_tokens_plan`: Max tokens for plan generation (default: 1024)
- `--max_tokens_solution`: Max tokens for solution generation (default: 2048)
- `--batch_size`: Batch size for generation (default: 8)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `--gpu_memory_utilization`: GPU memory utilization (default: 0.85)

## Expected Output

For N input problems, you'll get 4×N SFT samples (one for each plan/attempt).

Example output structure:
```
{
    'messages': [
        {'role': 'user', 'content': '...SAT problem...'},
        {'role': 'assistant', 'content': '...multi-plan response...'}
    ],
    'data_source': 'sat-3-5',
    'ability': 'math',
    'extra_info': {
        'attempt_num': 1,
        'plan_used': '...',
        'all_plans': ['...', '...', '...', '...'],
        ...original metadata...
    }
}
```

## Using the Generated Data for SFT

The output parquet file can be directly used with HuggingFace's SFT training:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer

# Load your generated SFT data
dataset = load_dataset('parquet', data_files='sft_data/output/sat_2to3_multi_plan_sft.parquet')

# Standard SFT training
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    dataset_text_field="messages",  # or process messages to text
    max_seq_length=4096,
)

trainer.train()
```

## Tips

1. **Start small**: Test with `--num_samples 10` first to verify the format
2. **Monitor GPU memory**: Adjust `--batch_size` and `--gpu_memory_utilization` based on your GPU
3. **Multiple GPUs**: Set `--tensor_parallel_size` to the number of GPUs available
4. **Quality check**: The script prints a sample at the end - verify it looks good
5. **Dataset variety**: Generate from both easy and hard problems for balanced training

## File Structure

```
sft_data/
├── generate_multi_plan_sft_data.py  # Main generation script
├── generate_sft_data.sh             # Convenience launch script
├── README.md                        # This file
└── output/                          # Generated SFT data will be saved here
    └── *.parquet
```

## Troubleshooting

**Out of memory errors:**
- Reduce `--batch_size`
- Reduce `--max_tokens_solution`
- Lower `--gpu_memory_utilization`

**Plans not being generated correctly:**
- Increase `--temperature` for more diverse plans
- Check the model is loaded correctly

**Slow generation:**
- Increase `--batch_size` if you have GPU memory
- Use `--tensor_parallel_size` for multi-GPU

## Next Steps

After generating SFT data:
1. Inspect a few samples to verify quality
2. Run SFT training on the generated data
3. Evaluate the fine-tuned model on your test set
4. Use the fine-tuned model as initialization for GRPO training


