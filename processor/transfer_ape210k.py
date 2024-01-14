'''
Author: Macielyoung
Date: 2023-11-16 10:15:49
Description: transfer ape210k dataset into multiturn chatting data format
'''
from datasets import load_dataset
from collections import defaultdict
import random


# def process_ape_dataset(example):
#     res = defaultdict(list)
    
#     questions = example['question_chinese']
#     equations = example['equation']
#     results = example['result_orig_format']

#     responses = ['对以上问题进行分析，可以使用以下等式来求解：\n{}\n最终解为：{}',
#                  '经过分析，我们可以用以下等式来解决该问题：\n{}\n最终答案为：{}']
#     for question, equation, result in zip(questions, equations, results):
#         res['query'].append(question)
#         response = random.choice(responses)
#         response = response.format(equation, result)
#         res['response'].append(response)
    
#     return res        

# ape210k_file = "../datasets/ape210k.jsonl"
# ape210k_dataset = load_dataset("json", data_files=ape210k_file)['train']
# print(ape210k_dataset)


# ape210k_dataset = ape210k_dataset.map(process_ape_dataset,
#                                       batched=True,
#                                       batch_size=300,
#                                       remove_columns=ape210k_dataset.column_names)
# print(ape210k_dataset)
# print(ape210k_dataset[0])


# ape210k_qa_file = "../datasets/ape210k_qa.jsonl"
# ape210k_dataset.to_json(ape210k_qa_file, force_ascii=False)


# # ape210k_dataset = ape210k_dataset.filter(lambda example: len(example['response']) > 50)
# # print(ape210k_dataset)
# # print(ape210k_dataset[0])


def process_ape210k_math_dataset(example):
    '''处理ape210k数学题数据集'''
    res = defaultdict(list)
    queries = example['query']
    responses = example['response']
    
    for query, response in zip(queries, responses):
        if len(query + response) > 1200:
            continue
        else:
            res['dialogue'].append([{"human": query, "assistant": response}])
    
    return res


ape210k_math_file = "../datasets/ape210k_qa.jsonl"
ape210k_math_dataset = load_dataset("json", data_files=ape210k_math_file)['train']
ape210k_math_dataset = ape210k_math_dataset.filter(lambda example: len(example['response']) > 50)
print(ape210k_math_dataset)


ape210k_math_dataset = ape210k_math_dataset.map(process_ape210k_math_dataset,
                                                    batched=True,
                                                    batch_size=300,
                                                    #   num_proc=10,
                                                    remove_columns=ape210k_math_dataset.column_names)
print(ape210k_math_dataset)