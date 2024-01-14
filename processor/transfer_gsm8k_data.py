'''
Author: Macielyoung
Date: 2023-11-03 15:07:20
Description: transfer gsm8k dataset into multiturn chatting data format
'''
from datasets import load_dataset, Dataset


def extract_language(prompt):
    instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request in English. Please answer in English."
    instruction_words = instruction.split(" ")
    prompt_words = prompt.split(" ")
    language = prompt_words[len(instruction_words) - 1].split(".")[0]
    return language
    

gsm8k_file = "../datasets/GSM8KInstruct_Cross.json"
gsm8k_dataset = load_dataset("json", data_files=gsm8k_file)['train']
print(gsm8k_dataset)

gsm8k_df = gsm8k_dataset.to_pandas()
gsm8k_df['language'] = gsm8k_df.apply(lambda x: extract_language(x['prompt']), axis=1)
gsm8k_df = gsm8k_df[gsm8k_df['language'].isin(['English', 'Chinese'])]

gsm8k_dataset = Dataset.from_pandas(gsm8k_df)
gsm8k_dataset = gsm8k_dataset.select_columns(['prompt', 'chosen', 'language'])
print(gsm8k_dataset)

gsm8k_math_file = "../datasets/GSM8KInstruct_Math.json"
gsm8k_dataset.to_json(gsm8k_math_file)