'''
Author: Macielyoung
Date: 2023-08-08 14:10:48
Description: merge lora weight into base model
'''
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from loguru import logger
import argparse
"""
使用该脚本，将lora的权重合并大base model中
"""


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_model", type=str, default="/root/pretrained/Qwen-14B-Chat", help="pretrained model path or id")
    parser.add_argument("--qlora_model", type=str, default="../models/qwen-14b-chat/v1/checkpoint-500", help="qlora model path")
    parser.add_argument("--save_path", type=str, default="../models/finetuned/qwen-14b-chat-v1-500-qlora", help="model saved path")
    args = parser.parse_args()
    return args


def merge_lora_to_base_model(base_model, 
                             qlora_model, 
                             save_path):
    # Baichuan-13B-Chat Merging Model
    # base_model = "baichuan-inc/Baichuan-13B-Chat"
    # qlora_model = "../models/baichuan/v1/checkpoint-500"
    # save_path = '../models/finetuned/baichuan-13b-chat-v1-500-qlora'
    
    # # Qwen-14B-Chat Merging Model
    # base_model = "/root/pretrained/Qwen-14B-Chat"
    # qlora_model = "../models/qwen-14b-chat/v1/checkpoint-500"
    # save_path = "../models/finetuned/qwen-14b-chat-v1-500-qlora"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model = PeftModel.from_pretrained(model, qlora_model)
    logger.info("load model done ...")
    model = model.merge_and_unload()
    logger.info("merge model done ...")

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    logger.info("save model and tokenizer done ...")


if __name__ == '__main__':
    args = parse_args()
    merge_lora_to_base_model(args.base_model,
                             args.qlora_model,
                             args.save_path)