import os
import sys
from typing import List

import torch
import transformers
from datasets import load_dataset, load_from_disk
from loguru import logger

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training
)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
from transformers import Trainer
import bitsandbytes as bnb
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from transformers.modeling_utils import unwrap_model


class QLoraTrainer(Trainer):
    
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info("saving qlora model to: {}".format(output_dir))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)
        
        model = unwrap_model(self.model)
        logger.info("saving embed_tokens and lm_head to {}".format(output_dir))
        torch.save(model.model.model.embed_tokens.state_dict(), os.path.join(output_dir, 'embed_tokens.bin'))
        torch.save(model.model.lm_head.state_dict(), os.path.join(output_dir, 'lm_head.bin'))
        

def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "../mistral_dialogues",
    output_dir: str = "../models/mistral-chinese-7b-instruct/v1/",
    # training hyperparams
    batch_size: int = 32,
    micro_batch_size: int = 2,
    num_epochs: int = 2,
    learning_rate: float = 5e-4,
    cutoff_len: int = 2048,
    val_set_size: int = 0,
    lr_scheduler_type: str = "cosine",
    use_gradient_checkpointing: bool = False,
    # lora hyperparams
    train_embedding: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "W_pack"
    ],
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
):
    logger.info(
        f"Training Mistral-Chinese-7B-Instruct-QLoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lr_scheduler_type: {lr_scheduler_type}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"group_by_length: {group_by_length}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='baichuan-inc/baichuan-7B'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    # device_map = {"": 0}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # load model in 4bit
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        trust_remote_code=True,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    # model.generation_config = GenerationConfig.from_pretrained(base_model)

    # load tokenizer that merge chinese tokens into raw english tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    logger.info("tokenizer len: {}".format(len(tokenizer)))
    # mistral tokenizer without pad and eos token
    tokenizer.pad_token = tokenizer.eos_token
    
    # if tokens num is not equal to embedding size, resize model embedding size
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        logger.info("resize the embedding size by the size of the tokenizer")
        model.resize_token_embeddings(len(tokenizer))
        # logger.info("model info: {}".format(model))

    def tokenize_dialogue(example):
        dialogues = example['dialogue']
        
        input_ids = []
        labels = []
        system_text = "### System:\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        system_ids = tokenizer.encode(system_text)
        input_ids += system_ids
        labels += len(system_ids) * [-100]
        for cid, dialogue in enumerate(dialogues):
            # chatting format: user: [user_text]\n\nassistant: [assistant_text][eos_token]
            human_text = dialogue['human']
            assistant_text = dialogue['assistant']
            
            human_text = "use: {}\n\nassistant: ".format(human_text)
            human_ids = tokenizer.encode(human_text)
            assistant_ids = tokenizer.encode(assistant_text)
            
            # 添加 user id, 用户问题前加入 user token id
            input_ids += human_ids
            labels += len(human_ids) * [-100]
            
            # 添加 assistant id，系统回复前加入 assistant token id，后面加入 eos token id
            input_ids += assistant_ids + [tokenizer.eos_token_id]
            labels += assistant_ids + [tokenizer.eos_token_id]
        
        assert len(input_ids) == len(labels)
        result = {
            'input_ids': input_ids,
            'labels': labels
        }
        return result

    def data_collator(features: list) -> dict:
        # cut off the input and label
        input_ids_list = [feature['input_ids'][:cutoff_len] for feature in features]
        labels_list = [feature['labels'][:cutoff_len] for feature in features]
        
        # pad token from left
        input_ids = pad_sequence([torch.tensor(input_ids[::-1]) for input_ids in input_ids_list], 
                                 batch_first=True,
                                 padding_value=tokenizer.pad_token_id).flip(dims=[1])
        labels = pad_sequence([torch.tensor(labels[::-1]) for labels in labels_list], 
                              batch_first=True, 
                              padding_value=-100).flip(dims=[1])
        
        input_ids = input_ids.long()
        labels = labels.long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id),
            "labels": labels,
        }

    if use_gradient_checkpointing:
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)
    else:
        model = prepare_model_for_int8_training(model)
    
    # add adapter modules for all linear layer
    lora_target_modules = find_all_linear_names(model)
    logger.info("lora target modules: {}".format(lora_target_modules))
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".jsonl"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_from_disk(data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    model.config.torch_dtype = torch.float32
    
    if train_embedding:
        # 针对增量中文词，需要训练embedding
        for n, p in model.named_parameters():
            if "embed_tokens" in n or "lm_head" in n:
                p.requires_grad = True
    
    logger.info("data info: {}".format(data))
    if val_set_size > 0:
        # split data into train and test dataset
        train_val = data['train'].train_test_split(
            test_size=val_set_size, shuffle=True, seed=215
        )
        train_data = (
            train_val['train'].shuffle().map(tokenize_dialogue, remove_columns=train_val['train'].column_names)
        )
        val_data = (
            train_val['test'].shuffle().map(tokenize_dialogue, remove_columns=train_val['test'].column_names)
        )
    else:
        train_data = data['train'].shuffle().map(tokenize_dialogue, remove_columns=data['train'].column_names)
        val_data = data['test'].shuffle().map(tokenize_dialogue, remove_columns=data['test'].column_names)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        
    train_args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=500,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        fp16=True,
        logging_steps=50,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="tensorboard"
    )

    trainer = QLoraTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=train_args,
        data_collator=data_collator,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    # model.save_pretrained(output_dir)
    trainer.save_model(output_dir)

    logger.info(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
    

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="../dialogues", help="sft data path")
    parser.add_argument("--pretrained", type=str, default="baichuan-inc/baichuan-7B", help="pretrained model from huggingface hub")
    parser.add_argument("--save_path", type=str, default="../models", help="model saved path")
    parser.add_argument("--epoches", type=int, default=5, help="epoch num")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="micro batch size")
    parser.add_argument("--val_set_size", type=int, default=2000, help="validation set size")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate num")
    parser.add_argument("--max_length", type=int, default=2048, help="sentence max length")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    # args = parse_args()
    # train(base_model=args.pretrained,
    #       data_path=args.data_path,
    #       output_dir=args.save_path,
    #       batch_size=args.batch_size,
    #       micro_batch_size=args.micro_batch_size,
    #       num_epochs=args.epoches,
    #       learning_rate=args.lr,
    #       cutoff_len=args.max_length,
    #       val_set_size=args.val_set_size,
    #       lora_r=args.lora_r,
    #       lora_alpha=args.lora_alpha,
    #       lora_dropout=args.lora_dropout,
    #       )
    # base_model = "baichuan-inc/Baichuan2-13B-Chat"
    base_model = "/root/pretrained/Mistral-7B-Chinese"
    train(base_model=base_model)