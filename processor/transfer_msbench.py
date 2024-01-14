'''
Author: Macielyoung
Date: 2023-11-24 14:33:19
Description: 
'''
from datasets import load_dataset
from collections import defaultdict


def generate_assistant_values(example):
    res = defaultdict(list)
    conversations = example['conversations']
    for conversation in conversations:
        assistant_turns = [turn['value'] for turn in conversation if turn['from'] == "assistant"]
        assistant_values = "".join(assistant_turns)
        res['assistant'].append(assistant_values)
    return res


def process_msagentbench_data(example):
    res = defaultdict(list)
    conversations = example['conversations']
    for conversation in conversations:
        # data format: [{'human': "", "assistant": ""} ...]
        rounds = []
        round_dict = {}
        for turn in conversation:
            from_field = turn['from']
            value_field = turn['value']
            if from_field == "system":
                continue
            elif from_field == "user":
                round_dict['human'] = value_field
            elif from_field == "assistant":
                round_dict['assistant'] = value_field
                rounds.append(round_dict)
                round_dict = {}
        res['dialogue'].append(rounds)
    return res

msbench_file = "../datasets/MSAgent-Bench.jsonl"
msbench_dataset = load_dataset('json', data_files=msbench_file)['train']

print(msbench_dataset)
print(msbench_dataset[0])

processed_msbench_dataset = msbench_dataset.map(generate_assistant_values, 
                                                batched=True,
                                                batch_size=100,
                                                num_proc=20,
                                                )
print(processed_msbench_dataset)
# print(processed_msbench_dataset[0])

processed_msbench_dataset = processed_msbench_dataset.filter(lambda example: "<|startofthink|>" not in example['assistant'] and "<|endofthink|>" not in example['assistant'])
print(processed_msbench_dataset)

processed_msbench_dataset = processed_msbench_dataset.map(process_msagentbench_data,
                                                          batched=True,
                                                          batch_size=100,
                                                          num_proc=20,
                                                          remove_columns=processed_msbench_dataset.column_names)
print(processed_msbench_dataset)
print(processed_msbench_dataset[0])

save_file = "../datasets/MSAgent_Dialogues.jsonl"
processed_msbench_dataset.to_json(save_file, force_ascii=False)