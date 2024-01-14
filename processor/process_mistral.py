'''
Author: Macielyoung
Date: 2023-06-17 14:33:54
Description: process dialogue for English LLM model
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

def process_openorca_dataset(example):
    '''
    处理openorca数据集
    '''
    res = defaultdict(list)
    instructions = example['instruction']
    inputs = example['input']
    outputs = example['output']
    for instruction, input_line, output_line in zip(instructions, inputs, outputs):
        # res['cid'].append(cid)
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

def process_platypus_dataset(example):
    '''
    处理platypus数据集
    '''
    res = defaultdict(list)
    instructions = example['instruction']
    inputs = example['input']
    outputs = example['output']
    for instruction, input_line, output_line in zip(instructions, inputs, outputs):
        # res['cid'].append(cid)
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

def process_lima_wizardlm_dataset(example):
    '''
    处理lima_wizardlm数据集
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
    
if __name__ == "__main__":
    # load moss dataset
    moss_file = "../datasets/moss-003-sft-data.jsonl"
    moss_dataset = load_dataset("json", data_files=moss_file)['train']
    moss_dataset = moss_dataset.map(process_moss_dataset,
                                    batched=True,
                                    batch_size=50,
                                    num_proc=10,
                                    remove_columns=moss_dataset.column_names)
    logger.info("load moss data done, info: {}".format(moss_dataset))
    # moss data num: 977604
    moss_dataset_num = len(moss_dataset)
    moss_sample_num = moss_dataset_num // 10 * 8
    moss_sample_list = random.sample(range(moss_dataset_num), moss_sample_num)
    moss_dataset = moss_dataset.select(moss_sample_list)
    logger.info("selected moss dataset info: {}".format(moss_dataset))
    logger.info("moss first line: {}".format(moss_dataset['dialogue'][0]))
    
    # load gsm8k math dataset
    gsm8k_math_file = "../datasets/GSM8KInstruct_Math.json"
    gsm8k_math_dataset = load_dataset("json", data_files=gsm8k_math_file)['train']
    gsm8k_math_dataset = gsm8k_math_dataset.map(process_gsm8k_math_dataset,
                                          batched=True,
                                          batch_size=300,
                                        #   num_proc=20,
                                          remove_columns=gsm8k_math_dataset.column_names)
    logger.info("load school math data done, info: {}".format(gsm8k_math_dataset))
    # gsm8k_math data num: 14837
    gsm8k_math_dataset_num = len(gsm8k_math_dataset)
    gsm8k_math_sample_num = gsm8k_math_dataset_num // 10 * 9
    gsm8k_math_sample_list = random.sample(range(gsm8k_math_dataset_num), gsm8k_math_sample_num)
    gsm8k_math_dataset = gsm8k_math_dataset.select(gsm8k_math_sample_list)
    logger.info("selected gsm8k math dataset info: {}".format(gsm8k_math_dataset))
    logger.info("gsm8k math first line: {}".format(gsm8k_math_dataset['dialogue'][0]))
    
    # load openorca dataset
    openorca_file = "../datasets/open_orca_chinese-2.json"
    openorca_dataset = load_dataset("json", data_files=openorca_file)['train']
    openorca_dataset = openorca_dataset.map(process_openorca_dataset,
                                            batched=True,
                                            batch_size=300,
                                            num_proc=15,
                                            remove_columns=openorca_dataset.column_names)
    openorca_dataset_num = len(openorca_dataset)
    logger.info("openorca_dataset_num: {}".format(openorca_dataset_num))
    openorca_sample_num = openorca_dataset_num // 10 * 8
    openorca_sample_list = random.sample(range(openorca_dataset_num), openorca_sample_num)
    openorca_dataset = openorca_dataset.select(openorca_sample_list)
    logger.info("openorca dataset info: {}".format(openorca_dataset))
    logger.info("openorca first line: {}".format(openorca_dataset['dialogue'][0]))
    
    # load platypus dataset
    platypus_file = "../datasets/open-platypus-chatgpt-chinese.json"
    platypus_dataset = load_dataset("json", data_files=platypus_file)['train']
    logger.info("platypus dataset info: {}".format(platypus_dataset))
    platypus_dataset = platypus_dataset.map(process_platypus_dataset,
                                            batched=True,
                                            batch_size=300,
                                            # num_proc=15,
                                            remove_columns=platypus_dataset.column_names)
    platypus_dataset_num = len(platypus_dataset)
    logger.info("platypus_dataset_num: {}".format(platypus_dataset_num))
    platypus_sample_num = platypus_dataset_num // 10 * 9
    platypus_sample_list = random.sample(range(platypus_dataset_num), platypus_sample_num)
    platypus_dataset = platypus_dataset.select(platypus_sample_list)
    logger.info("platypus dataset info: {}".format(platypus_dataset))
    logger.info("platypus first line: {}".format(platypus_dataset['dialogue'][0]))                            
    
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
    
    # load oaast dataset
    oaast_file = "../datasets/oaast_sft_dialogue_zh.jsonl"
    oaast_dataset = load_dataset("json", data_files=oaast_file)['train']
    # oaast_dataset = oaast_dataset.remove_columns('turn')
    oaast_dataset_num = len(oaast_dataset)
    logger.info("oaast_dataset_num: {}".format(oaast_dataset_num))
    oaast_sample_num = oaast_dataset_num // 10 * 9
    oaast_sample_list = random.sample(range(oaast_dataset_num), oaast_sample_num)
    oaast_dataset = oaast_dataset.select(oaast_sample_list)
    logger.info("oaast dataset info: {}".format(oaast_dataset))
    logger.info("oaast first line: {}".format(oaast_dataset['dialogue'][0]))
    
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
    
    # load lima dataset
    lima_wizardlm_file = "../datasets/LIMI-WizardLM_V2_everything.json"
    lima_wizardlm_dataset = load_dataset("json", data_files=lima_wizardlm_file)['train']
    logger.info("lima_wizardlm dataset info: {}".format(lima_wizardlm_dataset))
    lima_wizardlm_dataset = lima_wizardlm_dataset.map(process_lima_wizardlm_dataset,
                                                      batched=True,
                                                      batch_size=200,
                                                      remove_columns=lima_wizardlm_dataset.column_names)
    logger.info("lima_wizardlm dataset info: {}".format(lima_wizardlm_dataset))
    lima_wizardlm_dataset_num = len(lima_wizardlm_dataset)
    lima_wizardlm_sample_num = lima_wizardlm_dataset_num // 5 * 4
    lima_wizardlm_sample_list = random.sample(range(lima_wizardlm_dataset_num), lima_wizardlm_sample_num)
    lima_wizardlm_dataset = lima_wizardlm_dataset.select(lima_wizardlm_sample_list)
    logger.info("lima_wizardlm dataset info: {}".format(lima_wizardlm_dataset))
    logger.info("lima_wizardlm first line: {}".format(lima_wizardlm_dataset['dialogue'][0]))
    
    dialogue_dataset = concatenate_datasets([
                                            moss_dataset,
                                            gsm8k_math_dataset,
                                            openorca_dataset,
                                            platypus_dataset,
                                            alpaca_gpt4_dataset,
                                            oaast_dataset,
                                            sharegpt_dataset,
                                            lima_wizardlm_dataset,
                                            ])
    logger.info("merged dataset info: {}".format(dialogue_dataset))
    logger.info("dialogue data first line: {}".format(dialogue_dataset['dialogue'][0]))
    
    train_val_dataset = dialogue_dataset.train_test_split(
        test_size=30000, shuffle=True, seed=215
    )
    logger.info("train and eval dataset info: {}".format(train_val_dataset))
    
    dialogue_path = "../mistral_dialogues"
    train_val_dataset.save_to_disk(dialogue_path)