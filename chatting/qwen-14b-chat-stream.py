'''
Author: Macielyoung
Date: 2023-08-02 21:50:29
Description: qwen-14b-chat model with qlora adapter
'''
# import asyncio


from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
from transformers.generation.utils import GenerationConfig
import torch
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import readline


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_model", type=str, default="/root/pretrained/Qwen-14B-Chat", help="pretrained model path or id")
    parser.add_argument("--qlora_model", type=str, default="../models/qwen-14b-chat/v1/checkpoint-500", help="qlora model path")
    parser.add_argument("--is_multi_turn", type=int, default=1, help="support multi turn dialogue")
    args = parser.parse_args()
    return args


def build_chat_input(message, history):
    '''
    构建用户问答对话列表
    '''
    chatting_text = "### System:\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    if len(history) > 0:
        for pair in history:
            human = pair['query']
            assistant = pair['answer']
            # human, assistant = pair
            chatting_text += "user: {}\n\nassistant: {}{}".format(human, assistant, "</s>")
    chatting_text += "user: {}\n\nassistant: ".format(message)
    return chatting_text


def generate_streaming(tokenizer, model, text, streamer):
    text_ids = tokenizer.encode(text)
    input_ids = torch.tensor([text_ids]).to(model.device)
    outputs = model.generate(input_ids=input_ids, 
                             do_sample=True,
                             max_new_tokens=2048, 
                             top_p=0.75,
                             temperature=0.6,
                             repetition_penalty=1.2, 
                             eos_token_id=tokenizer.eos_token_id,
                             streamer=streamer)
    output = outputs[0][input_ids.shape[-1]: ]
    response = tokenizer.decode(output, skip_special_tokens=True)
    return response


if __name__ == "__main__":
    args = parse_args()
    
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.eos_token = "<|endoftext|>"

    # generate with streamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        fp16=True,
        # torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.qlora_model)
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(args.base_model)
    print("load model and tokenizer done!")

    history = []
    while True:
        text = input("please input your question:\n")
        if args.is_multi_turn:
            text = build_chat_input(text, history)
        response = generate_streaming(tokenizer, model, text, streamer)
        history.append({'query': text, 'answer': response})