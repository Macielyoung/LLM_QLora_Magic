'''
Author: Macielyoung
Date: 2023-08-12 15:50:28
Description: transfer sharegpt dataset into multiturn chatting data format
'''

from datasets import load_dataset
from collections import defaultdict
import json


def process_dialogue_dataset(example):
    '''
    保存sharegpt数据为多轮对话形式
    '''
    res = defaultdict(list)
    instructions = example['instruction']
    inputs = example['input']
    outputs = example['output']
    historys = example['history']
    
    for instruction, input_line, output_line, history in zip(instructions, inputs, outputs, historys):
        multiturn = []
        
        # 先处理历史对话
        for turn in history:
            human, assistant = turn
            history_item = {'human': human,
                            'assistant': assistant}
            multiturn.append(history_item)

        # 最后处理当前指令问答
        if input_line == "":
            human = instruction
        else:
            human = instruction + " " + input_line
        assistant = output_line
        local_item = {'human': human,
                      'assistant': assistant}
        multiturn.append(local_item)
        # turn_num = len(multiturn)
        
        res['dialogue'].append(multiturn)
        # res['turn'].append(turn_num)
    return res


def save_jsonl(dataset, save_file):
    with open(save_file, 'w', encoding='utf-8') as f:
        for example in dataset:
            json_example = json.dumps(example, ensure_ascii=False)
            f.write(json_example+"\n")
        

if __name__ == "__main__":
    sharegpt_file = "../datasets/sharegpt_zh_27k.json"
    # sharegpt_file = "../datasets/oaast_sft_zh.json"
    sharegpt_dataset = load_dataset('json', data_files=sharegpt_file)['train']
    
    sharegpt_dataset = sharegpt_dataset.map(process_dialogue_dataset, 
                                             batched=True,
                                             batch_size=50,
                                             num_proc=5,
                                             remove_columns=sharegpt_dataset.column_names)
    print(sharegpt_dataset)
    print(sharegpt_dataset[5])
    
    save_file = "../datasets/sharegpt_dialogue_zh_27k.jsonl"
    # save_file = "../datasets/oaast_sft_dialogue_zh.jsonl"
    save_jsonl(sharegpt_dataset, save_file)