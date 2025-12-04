import datasets

data_file = "/home/zhaoyiz/projects/data/apps_benign_prompt_short/test.parquet"
dataset = datasets.load_dataset("parquet", data_files=data_file)["train"]
print(dataset[0])