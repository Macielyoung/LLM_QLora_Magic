'''
Author: Macielyoung
Date: 2023-06-30 13:21:43
Description: preprocess text for pretrainer script
'''
from datasets import load_dataset, Dataset, concatenate_datasets
from collections import defaultdict
from loguru import logger
import json


Min_Sentence_Len = 50
Max_Sentence_Len = 1500


def process_news_dataset(example):
    '''
    处理新闻预训练文本语料数据集
    '''
    res = defaultdict(list)
    paragraphs = example['段落']
    for item in paragraphs:
        for paragraph in item:
            text = paragraph['内容']
            if len(text) < Min_Sentence_Len or len(text) > Max_Sentence_Len:
                continue
            res['text'].append(text)
    return res


def process_report_dataset(example):
    '''
    处理公司年报预训练文本语料数据集
    '''
    res = defaultdict(list)
    paragraphs = example['text']
    for item in paragraphs:
        item = json.loads(item)
        for paragraph in item:
            text = paragraph['内容']
            if len(text) < Min_Sentence_Len or len(text) > Max_Sentence_Len:
                continue
            res['text'].append(text)
    return res


def process_wiki_dataset(example):
    '''
    处理维基百科数据集
    '''
    res = defaultdict(list)
    completions = example['completion']
    for completion in completions:
        if len(completion) < Min_Sentence_Len or len(completion) > Max_Sentence_Len:
            continue
        res['text'].append(completion)
    return res
    

if __name__ == "__main__":
    # news_dataset = Dataset.from_list([])
    
    # news_ids = 1
    # for news_id in range(news_ids):
    #     news_id_file = "../corpus/company_report/{}.jsonl".format(news_id)
    #     news_id_dataset = load_dataset("json", data_files=news_id_file)['train']
    #     news_id_dataset = news_id_dataset.map(process_report_dataset,
    #                                           batched=True,
    #                                           batch_size=50,
    #                                           num_proc=20,
    #                                           remove_columns=news_id_dataset.column_names)
    #     logger.info("news id: {}, dataset info: {}".format(news_id, news_id_dataset))
    #     news_dataset = concatenate_datasets([news_dataset, news_id_dataset])
    
    # logger.info("dataset first line: {}".format(news_dataset[0]))
    
    # train_val_dataset = news_dataset.train_test_split(
    #     test_size=20000, shuffle=True, seed=215
    # )
    # logger.info("train and eval dataset info: {}".format(train_val_dataset))
    
    # dialogue_path = "../news_corpus"
    # train_val_dataset.save_to_disk(dialogue_path)
    
    wiki_file = "../corpus/wiki/wikipedia-cn-20230720-filtered.json"
    wiki_dataset = load_dataset("json", data_files=wiki_file)['train']
    wiki_dataset = wiki_dataset.map(process_wiki_dataset,
                                    batched=True,
                                    batch_size=50,
                                    num_proc=20,
                                    remove_columns=wiki_dataset.column_names)
    logger.info("wiki_dataset first line: {}".format(wiki_dataset[0]))
    
    train_val_dataset = wiki_dataset.train_test_split(
        test_size=10000, shuffle=True, seed=215
    )
    logger.info("train and eval dataset info: {}".format(train_val_dataset))
    
    dialogue_path = "../wiki_corpus"
    train_val_dataset.save_to_disk(dialogue_path)