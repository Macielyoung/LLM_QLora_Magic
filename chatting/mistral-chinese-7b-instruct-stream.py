'''
Author: Macielyoung
Date: 2023-08-02 21:50:29
Description: mistral-7b model with chinese extension and qlora adapter
'''
# import asyncio


from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.generation.utils import GenerationConfig
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import readline

base_model = "/root/pretrained/Mistral-7B-Chinese"
qlora_model = "../models/mistral-chinese-7b-instruct/v1/checkpoint-300"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
print("tokenizer len: {}".format(len(tokenizer)))
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# generate with streamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    # trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) != embedding_size:
    print("resize the embedding size: {} -> {}".format(embedding_size, len(tokenizer)))
    model.resize_token_embeddings(len(tokenizer))
    
embed_tokens_file = os.path.join(qlora_model, 'embed_tokens.bin')
lm_head_file = os.path.join(qlora_model, 'lm_head.bin')
if os.path.exists(embed_tokens_file) and os.path.exists(lm_head_file):
    print('Update embed_tokens and lm_head ...')
    embed_tokens_params = torch.load(embed_tokens_file)
    lm_head_params = torch.load(lm_head_file)

    model.model.embed_tokens.load_state_dict(embed_tokens_params)
    model.lm_head.load_state_dict(lm_head_params)
    print("model info: {}".format(model))
else:
    print('There are no embed_tokens and lm_head, we will not update the embed_tokens and lm_head')
    exit()

model = PeftModel.from_pretrained(model, qlora_model)
model.eval()
# model.generation_config = GenerationConfig.from_pretrained(base_model)
print("load model and tokenizer done!")


def generate_streaming(text, streamer):
    text = """### System:\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nuser: {}\n\nassistant: """.format(text)
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


while True:
    text = input("please input your question:\n")
    response = generate_streaming(text, streamer)
    # print("answer: {}".format(response))