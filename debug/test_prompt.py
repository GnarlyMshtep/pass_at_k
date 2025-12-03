from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
# prompt = "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant who can solve Python programming problems. Think step by step and provide complete, runnable solutions.\nuser\nYou will be shown a Python programming task. Your goal is to write a solution to the problem. Your solution should meet the following criteria:\n\nCriteria for code output:\n\n- Valid solution: The solution must pass all tests listed in the problem description, as well as some held-out tests you won't see.\n\nCriteria for thinking output:\n\n- Try very hard to come up with a working solution. These are hard problems, so you'll need to think carefully.\n\n- You might need to iterate on each different approach several times.\n\nOutput format:\n\n- In <answer> tags, output a COMPLETE, RUNNABLE Python program that reads input from stdin (using input()), solves the problem, and prints the result to stdout (using print()). Do NOT just provide a function definition - provide the entire working program including the input/output code. The program will be executed directly with test inputs passed via stdin.\n\nDon't be biased towards thinking you've done a good job, you might've made a mistake. That's OK, just try again!\n\nYou are given two arrays of integers a and b. For each element of the second array b_{j} you should find the number of elements in array a that are less than or equal to the value b_{j}.\n\n\n-----Input-----\n\nThe first line contains two integers n, m (1 ≤ n, m ≤ 2·10^5) — the sizes of arrays a and b.\n\nThe second line contains n integers — the elements of array a ( - 10^9 ≤ a_{i} ≤ 10^9).\n\nThe third line contains m integers — the elements of array b ( - 10^9 ≤ b_{j} ≤ 10^9).\n\n\n-----Output-----\n\nPrint m integers, separated by spaces: the j-th of which is equal to the number of such elements in array a that are less than or equal to the value b_{j}.\n\n\n-----Examples-----\nInput\n5 4\n1 3 5 7 9\n6 4 2 8\n\nOutput\n3 2 1 4\n\nInput\n5 5\n1 2 1 2 5\n3 1 4 1 5\n\nOutput\n4 2 4 2 5\nassistant\n"

prompt = '''
Write a python code to solve the following problem.
-----Input-----\n\nThe first line contains two integers n, m (1 ≤ n, m ≤ 2·10^5) — the sizes of arrays a and b.\n\nThe second line contains n integers — the elements of array a ( - 10^9 ≤ a_{i} ≤ 10^9).\n\nThe third line contains m integers — the elements of array b ( - 10^9 ≤ b_{j} ≤ 10^9).\n\n\n-----Output-----\n\nPrint m integers, separated by spaces: the j-th of which is equal to the number of such elements in array a that are less than or equal to the value b_{j}.\n\n\n-----Examples-----\nInput\n5 4\n1 3 5 7 9\n6 4 2 8\n\nOutput\n3 2 1 4\n\nInput\n5 5\n1 2 1 2 5\n3 1 4 1 5\n\nOutput\n4 2 4 2 5
Time and space complexity analyses are not needed. Let's think step by step and put the code at the end of the response.
'''

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)