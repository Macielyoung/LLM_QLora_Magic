'''
Author: Macielyoung
Date: 2023-08-02 21:50:29
Description: baichuan-13b-chat model with qlora adapter prediction
'''
# import asyncio


from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import readline

base_model = "/root/pretrained/Baichuan-13B-Chat"
qlora_model = "../models/baichuan-13b-chat/v1/checkpoint-100"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# generate with streamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map={"": 1}
)
model = PeftModel.from_pretrained(model, qlora_model)
model.eval()
model = model.to(device)
print("load model and tokenizer done!")


def generate_streaming(text, streamer, device):
    text = """### System:\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nuser: {}\n\nassistant: """.format(text)
    text_ids = tokenizer.encode(text)
    input_ids = [model.generation_config.user_token_id] + text_ids + [model.generation_config.assistant_token_id]
    input_ids = torch.tensor([input_ids]).to(device)
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
    response = generate_streaming(text, streamer, device)