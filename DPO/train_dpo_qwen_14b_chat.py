# 0. imports
import os
from typing import Dict
from typing import List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
import bitsandbytes as bnb
from trl import DPOTrainer
from loguru import logger


def get_gsm8k_paired(
    data_file: str = "../datasets/GSM8KInstruct_Math.json",
    sanity_check: bool = False,
    num_proc=10,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "### 系统:\n你是九方财富开发的AI助手，九章大模型。尽可能安全的回答我的问题，不要回复非法内容\n\n用户：" + <prompt> + "\n\n九章："
    """
    dataset = load_dataset(
        "json",
        data_files=data_file
    )['train']
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["### 系统:\n你是九方财富开发的AI助手，九章大模型。尽可能安全的回答我的问题，不要回复非法内容\n\n用户：{}\n\n九章：".format(prompt) for prompt in samples["prompt"]],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


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
    # model and data path
    model_name_or_path: str = "../models/finetuned/qwen-14b-chat-v4-6500-qlora",
    data_file: str = "../datasets/GSM8KInstruct_Math.json",
    output_dir: str = "../models/dpo/qwen-14b-chat-v4-6500-qlora/v2/",
    
    # training hyperparams
    beta: float = 0.1,
    gradient_accumulation_steps: int = 16,
    micro_batch_size: int = 1,
    num_epochs: int = 2,
    save_total_limit: int = 3,
    learning_rate: float = 5e-6,
    max_prompt_length: int = 1024,
    max_length: int = 1600,
    val_set_size: int = 1000,
    lr_scheduler_type: str = "cosine",
    optimizer_type: str = "paged_adamw_32bit",
    gradient_checkpointing: bool = True,
    warmup_steps: int = 100,
    logging_steps: int = 50,
    save_steps: int = 300,
    eval_steps: int = 300,
    report_to: str = "tensorboard",
    
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "W_pack"
    ],
):
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        
    # 1. load a pretrained model
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        load_in_4bit=True,
        quantization_config=quantization_config
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    logger.info("load sft model done!")
    
    lora_target_modules = find_all_linear_names(model)
    logger.info("target modules: {}".format(lora_target_modules))
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32
    
    # load ref model
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        load_in_4bit=True,
        quantization_config=quantization_config
    )
    logger.info("load ref model done!")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    # qwen tokenizer without pad and eos token
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|endoftext|>"
    
    # 2. Load the GSM8k paired dataset
    data_file = "../datasets/GSM8KInstruct_Math.json"
    gsm8k_dataset = get_gsm8k_paired(data_file)
    gsm8k_dataset = gsm8k_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length
        and len(x["prompt"]) + len(x["rejected"]) <= max_length
    )
    train_val_dataset = gsm8k_dataset.train_test_split(
        test_size=val_set_size, shuffle=True, seed=215
    )
    train_dataset = train_val_dataset['train']
    eval_dataset = train_val_dataset['test']
    logger.info("load dataset done!")
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    # 3. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        num_train_epochs=num_epochs,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        output_dir=output_dir,
        report_to=report_to,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        optim=optimizer_type,
        # bf16=True,
        fp16=True,
        save_total_limit=save_total_limit,
        remove_unused_columns=False,
        # ddp_find_unused_parameters=None,
        ddp_find_unused_parameters=False if ddp else None,
    )
    
    # 4. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # peft_config=peft_config,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
    )

    # 5. train
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    # 6. save
    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    

if __name__ == "__main__":
    model_name_or_path = "../models/finetuned/qwen-14b-chat-v4-6500-qlora"
    # model_name_or_path = "/root/pretrained/Qwen-7B-Chat"
    train(model_name_or_path)
