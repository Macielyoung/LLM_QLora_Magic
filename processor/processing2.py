'''
Author: Macielyoung
Date: 2023-06-17 14:33:54
Description: process dialogue for Chinese LLM model
'''
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
import random
from loguru import logger

MAX_SENTENCE_LENGTH = 1500

def process_moss_dataset(example):
    '''处理moss sft3数据集'''
    res = defaultdict(list)
    conversations = example['conversation']
    
    for conversation in conversations:
        dialogue = ""
        for turn in conversation:
            human = turn['human']
            human = human.replace("MOSS", "AI小助手").replace("Moss", "AI小助手")
            turn['human'] = human
            
            assistant = turn['assistant']
            assistant = assistant.replace("MOSS", "AI小助手").replace("Moss", "AI小助手")
            turn['assistant'] = assistant
            dialogue += human + assistant
            
        # res['cid'].append(cid)
        if len(dialogue) > MAX_SENTENCE_LENGTH:
            if len(conversation[0]['human'] + conversation[0]['assistant']) > MAX_SENTENCE_LENGTH:
                continue
            res['dialogue'].append([conversation[0]])
        else:
            res['dialogue'].append(conversation)
    return res

def process_gsm8k_math_dataset(example):
    '''
    处理GSM8k数学题数据集
    原始数据来源：https://huggingface.co/datasets/Mathoctopus/MGSM8KInstruct_Cross
    '''
    res = defaultdict(list)
    inputs = example['prompt']
    outputs = example['chosen']
    languages = example['language']
    
    for input_line, output_line, language in zip(inputs, outputs, languages):
        # res['cid'].append(cid)
        input_len = len(input_line.split(" "))
        if language == "English":
            # answer with English
            output_len = len(output_line.split(" "))
        else:
            # answer with Chinese
            output_len = len(output_line)
        
        if input_len + output_len > MAX_SENTENCE_LENGTH:
            continue
        else:
            res['dialogue'].append([{"human": input_line, "assistant": output_line}])
    
    return res

def process_alpaca_gpt4_dataset(example):
    '''
    处理alpaca_gpt4数据集
    '''
    res = defaultdict(list)
    instructions = example['instruction']
    inputs = example['input']
    outputs = example['output']
    for instruction, input_line, output_line in zip(instructions, inputs, outputs):
        if input_line == "":
            prompt = instruction
        else:
            prompt = instruction + input_line
        dialogue = prompt + " " + output_line
        if len(dialogue) > MAX_SENTENCE_LENGTH:
            continue
        else:
            res['dialogue'].append([{"human": prompt, "assistant": output_line}])
    
    return res

def process_sharegpt_dataset(example):
    '''
    处理sharegpt数据集
    '''
    res = defaultdict(list)
    diagloues = example['dialogue']
    for dialogue in diagloues:
        prompt = ""
        for pair in dialogue:
            human = pair['human']
            assistant = pair['assistant']
            prompt += human + " " + assistant + " "
        if len(prompt) <= MAX_SENTENCE_LENGTH:
            res['dialogue'].append(dialogue)
    
    return res


if __name__ == "__main__":
    # load gsm8k math dataset
    gsm8k_math_file = "../datasets/GSM8KInstruct_Math.json"
    gsm8k_math_dataset = load_dataset("json", data_files=gsm8k_math_file)['train']
    gsm8k_math_dataset = gsm8k_math_dataset.map(process_gsm8k_math_dataset,
                                          batched=True,
                                          batch_size=300,
                                        #   num_proc=20,
                                          remove_columns=gsm8k_math_dataset.column_names)
    logger.info("load gsm8k math data done, info: {}".format(gsm8k_math_dataset))
    # gsm8k_math data num: 14837
    gsm8k_math_dataset_num = len(gsm8k_math_dataset)
    gsm8k_math_sample_num = gsm8k_math_dataset_num // 10 * 9
    gsm8k_math_sample_list = random.sample(range(gsm8k_math_dataset_num), gsm8k_math_sample_num)
    gsm8k_math_dataset = gsm8k_math_dataset.select(gsm8k_math_sample_list)
    logger.info("selected gsm8k math dataset info: {}".format(gsm8k_math_dataset))
    logger.info("gsm8k math first line: {}".format(gsm8k_math_dataset['dialogue'][0]))
    
    # load alpaca_gpt4 dataset
    alpaca_gpt4_file = "../datasets/alpaca_gpt4_data_zh.json"
    alpaca_gpt4_dataset = load_dataset("json", data_files=alpaca_gpt4_file)['train']
    alpaca_gpt4_dataset = alpaca_gpt4_dataset.map(process_alpaca_gpt4_dataset,
                                            batched=True,
                                            batch_size=300,
                                            # num_proc=15,
                                            remove_columns=alpaca_gpt4_dataset.column_names)
    alpaca_gpt4_dataset_num = len(alpaca_gpt4_dataset)
    logger.info("alpaca_gpt4_dataset_num: {}".format(alpaca_gpt4_dataset_num))
    alpaca_gpt4_sample_num = alpaca_gpt4_dataset_num // 10 * 9
    alpaca_gpt4_sample_list = random.sample(range(alpaca_gpt4_dataset_num), alpaca_gpt4_sample_num)
    alpaca_gpt4_dataset = alpaca_gpt4_dataset.select(alpaca_gpt4_sample_list)
    logger.info("alpaca_gpt4 dataset info: {}".format(alpaca_gpt4_dataset))
    logger.info("alpaca_gpt4 first line: {}".format(alpaca_gpt4_dataset['dialogue'][0]))
    
    # load sharegpt dataset
    sharegpt_file = "../datasets/sharegpt_dialogue_zh_27k.jsonl"
    sharegpt_dataset = load_dataset("json", data_files=sharegpt_file)['train']
    sharegpt_dataset = sharegpt_dataset.map(process_sharegpt_dataset,
                                            batched=True,
                                            batch_size=300,
                                            remove_columns=sharegpt_dataset.column_names)
    sharegpt_dataset_num = len(sharegpt_dataset)
    logger.info("sharegpt_dataset_num: {}".format(sharegpt_dataset_num))
    sharegpt_sample_num = sharegpt_dataset_num // 5 * 4
    sharegpt_sample_list = random.sample(range(sharegpt_dataset_num), sharegpt_sample_num)
    sharegpt_dataset = sharegpt_dataset.select(sharegpt_sample_list)
    logger.info("sharegpt dataset info: {}".format(sharegpt_dataset))
    logger.info("sharegpt first line: {}".format(sharegpt_dataset['dialogue'][0]))
    
    dialogue_dataset = concatenate_datasets([
                                            gsm8k_math_dataset,
                                            alpaca_gpt4_dataset,
                                            sharegpt_dataset,
                        
                                            ])
    logger.info("merged dataset info: {}".format(dialogue_dataset))
    logger.info("dialogue data first line: {}".format(dialogue_dataset['dialogue'][0]))
    
    train_val_dataset = dialogue_dataset.train_test_split(
        test_size=3000, shuffle=True, seed=215
    )
    logger.info("train and eval dataset info: {}".format(train_val_dataset))
    
    dialogue_path = "../dialogues"
    train_val_dataset.save_to_disk(dialogue_path)