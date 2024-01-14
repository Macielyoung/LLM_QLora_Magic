## LLM_Magic

### 项目介绍

本项目主要介绍使用高效参数微调方法**(QLora)**对现有开源大模型微调的流程。

* PT（继续预训练）
* SFT（监督微调）
* DPO（直接偏好优化）
* Merging（adapter合并）
* Server（模型应用服务化）



### 常见数据集

<!DOCTYPE html>
<html>
<body>
  	<details>
    <summary>预训练数据集</summary>
      <p><a href="https://huggingface.co/datasets/olm/olm-wikipedia-20221220">Wikipedia(zh)</a></p>
      <p><a href="https://huggingface.co/datasets/Skywork/SkyPile-150B">SkyPile(zh)</a></p>
      <p><a href="https://huggingface.co/datasets/CASIA-LM/ChineseWebText">ChineseWebText(zh)</a></p>
</details>
</body>
</html>

<!DOCTYPE html>
<html>
<body>
  	<details>
    <summary>指令微调数据集</summary>
      <p><a href="https://huggingface.co/datasets/YeungNLP/moss-003-sft-data">moss(zh)</a></p>
      <p><a href="https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chinese-instruction-collection">alpaca_gpt4(zh)</a></p>
      <p><a href="https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/blob/main/Chinese-instruction-collection/sharegpt_zh_38K.json">sharegpt(zh)</a></p>
      <p><a href="https://huggingface.co/datasets/Mathoctopus/MGSM8KInstruct_Cross">gsm8k_math(multilingual)</a></p>
      <p><a href="https://huggingface.co/datasets/Azure99/blossom-math-v1">blossom_math(zh)</a></p>
      <p><a href="https://huggingface.co/datasets/Azure99/blossom-orca-v1">blossom_orca(zh)</a></p>
      <p><a href="https://huggingface.co/datasets/BAAI/COIG/blob/main/leetcode_instructions.jsonl">leetcode(zh)</a></p>
</details>
</body>
</html>

<!DOCTYPE html>
<html>
<body>
  	<details>
    <summary>偏好数据集</summary>
      <p><a href="https://huggingface.co/datasets/Mathoctopus/MGSM8KInstruct_Cross">gsm8k_math(multilingual)</a></p>
      <p><a href="https://huggingface.co/datasets/lvwerra/stack-exchange-paired">stack_exchange(en)</a></p>
      <p><a href="https://github.com/HIT-SCIR/huozi/blob/main/data/huozi_rlhf_data.csv">huozi_rlhf(zn)</a></p>
</details>
</body>
</html>



### 常见大模型

| 模型名                                                       | 模型大小 | 默认微调模块  | 基础语言 |
| ------------------------------------------------------------ | -------- | ------------- | -------- |
| [Baichuan](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | 7B,13B   | W_pack        | 中文     |
| [Qwen](https://huggingface.co/Qwen/Qwen-14B-Chat)            | 7B,14B   | c_attn        | 中文     |
| [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | 7B       | q_proj,v_proj | 英文     |
| [zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | 7B       | q_proj,v_proj | 英文     |



### CPT(继续预训练)

继续预训练主要是针对领域属性较强、语料较丰富的场景（如金融、医疗、法律等领域）。可能在这些领域中基模型相关知识较少，需要继续预训练来增加相应知识。

继续预训练需要的数据格式以纯文本为主，以下面数据为例。

```json
{
  "text": "转自：第一财经 9月23日，华商基金管理有限公司发布关于华商鸿源三个月定期开放纯债债券型证券投资基金的分红公告，该只基金将迎来2023年度第二次分红，收益分配基准日为9月19日。 根据公告披露，华商鸿源三个月定开纯债债券（014076.OF）基准日净值为1.0151元，可分红总额为70,194,703.13元，据此给出的基金分红方案为每10份基金份额派0.0510元。 与此同时，上述基金的权益登记日为9月26日，这将意味着当日持有基金份额的基民都可享受本次分红。其中，场外基金份额的除息日为9月26日，现金红利发放日为9月28日，届时请基金持有人注意持仓份额和净值变化。 另值得一提的是，选择红利再投资方式的投资者红利再投资所得的基金份额将按2023年9月26日的基金份额净值计算确定，本公司将于2023年9月27日对红利再投资的基金份额进行确认并通知各销售机构，本次红利再投资所得份额的持有期限自2023年9月27日开始计算，2023年9月28日起投资者可以查询。本基金采取在封闭期内封闭运作、封闭期与封闭期之间定期开放的运作方式。本基金自2023年8月31日15:00后，暂停申购、赎回、转换业务，进入下一个封闭运作管理期，本基金以三个月为一个封闭运作周期。 文章作者 灵犀"
}
```

以baichuan-13b-base模型为例，继续预训练步骤如下：

```shell
# 预处理纯文本
python preprocess.py(处理数据保存到磁盘上)

# 单节点多GPU运行
deepspeed --include=localhost:0,1,2,3 pretrain_bc_qlora.py --deepspeed_config ds.json 
```



### SFT(监督微调)

监督微调主要是在基模型或者对话模型基础上，利用指令微调问答数据集来对齐，实现多轮对话。

监督微调需要的数据格式主要是多轮指令问答数据，以下面数据为例。

```json
[ 
  { 
    "human": "量子通量作为设计" ,
    "assistant": "量子通量是量子力学中涉及到量子系统中粒子或能量的流动的概念。在设计的语境下，它可能指的是使用量子力学的原理来指导系统或产品的设计。然而，在设计领域中，这个术语并不常见，它的含义可能根据具体的语境而异。"
  }, 
  { 
    "human": "在网页和应用软件工程和设计的背景下",
    "assistant": "我不知道在Web和App软件工程和设计的背景下是否有任何关于“量子通量”的具体用法。一般来说，软件工程和设计涉及使用已建立的设计模式和原则，创建高效和用户友好的软件。量子力学是研究物质和能量在原子和亚原子级别上的行为的物理学分支，通常不在Web和App开发的背景下使用。它可能被用于量子计算的背景下，量子计算是一个相对较新的领域，利用诸如叠加和纠缠等量子力学现象对数据进行操作。但是，该领域仍处于起步阶段，目前还不清楚它如何在Web和App软件工程和设计的背景下使用。" 
  }, 
  { 
    "human": "弥补它" ,
    "assistant": "可以想象一种情境，即将量子力学的原理，如叠加和纠缠应用于网络和应用软件的设计中。例如，使用量子计算技术，可能可以创建能够同时存在于多种状态的网络和应用界面，使用户能够同时探索多种选项和可能性。此外，纠缠的概念可以用于设计高度互联的网络和应用界面，使用户能够轻松访问相关信息和内容。\n另一个例子可以在安全领域，量子密码学的原则可以应用于创建更安全、更私密的通信渠道，用于网络和应用界面。\n请记住，这些只是量子力学原理如何在网络和应用软件工程设计中应用的例子，这个领域仍处于起步阶段。"
  } 
]
```



#### 中文模型微调

通常中文大语言模型(如千问、百川等)，我们可以直接进行监督微调，考虑到资源，我们这里主要使用高效参数微调的方法（[QLora](https://github.com/artidoro/qlora)）微调模型。

以qwen_14B_chat模型为例，经过以下步骤即可训练。

```shell
# 如果有自己的SFT数据，最终处理上述的多轮对话数据格式即可。
python processing.py(处理数据保存到磁盘上)

# 单机多卡
deepspeed --include=localhost:0,1,2,3 train_qwen_14b_chat_qlora.py --deepspeed_config ds.json 
```



#### 英文模型微调

还有一些非常优秀的英文大语言模型(如llama系列，mistral等)，这些模型主要以英文为主，实现了出色的英文处理能力。同时，这些模型也支持中文解码，为中文用户提供了便利。然而，由于这些模型的原生中文词汇较少，导致在中文解码过程中，一个中文汉字可能会被解码为2-3个unicode token，从而影响解码的效率和上下文能力。因此针对这些英文基座模型进行监督微调前，需要进行中文词汇扩展合并，提高中文解码效率。

下面我们以mistral模型为例，展示中文词汇扩展和监督微调的流程。中文词表可以自己准备纯文本利用[sentencepiece](https://github.com/google/sentencepiece)训练，也可以直接利用现有开源中文模型的词表。我们直接使用现有开源中文模型的词表，比如baichuan系列模型。

* 下载baichuan模型的词表文件，[参考链接](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/tokenizer.model)

* 合并词表

  ```shell
  python merge_tokenizer.py 
  		--llama_tokenizer_dir your_mistral_model_path 
  		--chinese_sp_model_file your_baichuan_tokenizer_path
  ```

  --llama_tokenizer_dir：mistral模型本地路径或者huggingface id

  --chinese_sp_model_file：中文词表本地路径

* 使用新生成的词表文件代替原始词表文件

对mistral模型和baichuan-13b-chat模型词表合并，原始token为32000个，合并后有75449个。以下是一条测试文本，可以看出原始句子编码中部分中文词（植）需要三个token表示，合并后可以直接表示为一个token（植物）。

```json
{
	"测试文本": "今天天气真好，可以去植物园秋游了。The weather is wonderful today, providing an ideal opportunity for a leisurely autumn stroll in the botanical garden.",
	"mistral_tokens": ['▁', '今', '天', '天', '气', '真', '好', '，', '可', '以', '去', '<0xE6>', '<0xA4>', '<0x8D>', '物', '园', '秋', '游', '了', '。', 'The', '▁weather', '▁is', '▁wonderful', '▁today', ',', '▁providing', '▁an', '▁ideal', '▁opportunity', '▁for', '▁a', '▁le', 'isure', 'ly', '▁autumn', '▁st', 'roll', '▁in', '▁the', '▁bot', 'an', 'ical', '▁garden', '.'],
	"merged_mistral_tokens": ['▁今天', '天气', '真', '好', '，', '可以', '去', '植物', '园', '秋', '游', '了', '。', 'The', '▁w', 'eat', 'her', '▁is', '▁wonderful', '▁today', ',', '▁providing', '▁an', '▁ideal', '▁opportunity', '▁for', '▁a', '▁leisure', 'ly', '▁autumn', '▁stroll', '▁in', '▁the', '▁bot', 'anical', '▁garden', '.']
}
```

由于新增了部分中文相关token，因此监督微调过程中除了微调注意力相关层以外，还需要训练embedding和lm_head层，并保存相应权重。

```shell
# 如果有自己的SFT数据，最终处理上述的多轮对话数据格式即可。
python process_mistral.py(处理数据保存到磁盘上)

# 单机多卡
deepspeed --include=localhost:0,1,2,3 train_mistral_7b_instruct_qlora.py --deepspeed_config ds.json 
```



### DPO

直接偏好优化是在监督微调之后，进一步去拟合用户的偏好习惯。

偏好优化所需要的数据形式以下面数据为例。

```json
{
  "prompt": "给出一个数列，对其进行排序，要求输出从小到大排列的结果。\n\n一个待排序的数字序列，如[3, 1, 4, 1, 5, 9, 2, 6]\n",
  "chosen": "可以使用Python内置的sorted()函数进行排序，代码如下：\n\n```\nnum_list = [3, 1, 4, 1, 5, 9, 2, 6]\nsorted_list = sorted(num_list)\nprint(sorted_list)\n```\n\n输出结果为：[1, 1, 2, 3, 4, 5, 6, 9]",
  "rejected": "[1, 1, 2, 3, 4, 5, 6, 9]"
}
```

以qwen-14b-chat模型为例，优化监督微调模型流程如下：

```shell
deepspeed --include=localhost:0,1,2,3 train_dpo_qwen_14b_chat.py --deepspeed_config ds.json 
```



### Merging

使用Qlora训练中保存的checkpoint主要是adapter层变量，最终还需要和基础模型进行权重合并，以下是权重合并脚本。

```shell
python merge_qlora.py 
		--base_model base_model_path 
		--qlora_model your_qlora_model_saved_path 
		--save_path merged_model_saved_path
```

--base_model：基础模型路径或ID

--qlora_model：qlora模型权重路径

--save_path：合并模型保存路径



### Server

#### API Demo

这部分主要提供模型服务化部署，这里我们以[vllm框架](https://github.com/vllm-project/vllm)来部署我们的模型，vllm支持的模型结构可以查看这里：[支持模型结构](https://docs.vllm.ai/en/latest/models/supported_models.html)

我们以合并好的qwen模型为例，通过以下脚本启动服务：

```shell
CUDA_VISIBLE_DEVICES=0,1 python vllm_qwen_server.py 
		--model your_qwen_qlora_model 
		--trust-remote-code 
		--tokenizer-mode auto 
		--max-num-batched-tokens 10000 
		--tensor-parallel-size 2 
		--port 8080 
		--host 0.0.0.0
```

部署好服务后，即可使用curl或者requests进行请求，支持流式和非流式数据请求，以下是请求用例。

```shell
# 流式
curl --location 'http://you r_ip:port/ai/llm/generation_stream' \
--header 'Content-Type: application/json' \
--data '{
    "message": "你是谁",
    "history": [],
    "n": 1,
    "temperature": 0.3,
    "max_tokens": 8192,
    "stream": true,
    "stop_token_ids": [151643],
    "frequency_penalty": 1.2
}'

# 非流式
curl --location 'http://you r_ip:port/ai/llm/generation_stream' \
--header 'Content-Type: application/json' \
--data '{
    "message": "你是谁",
    "history": [],
    "n": 1,
    "temperature": 0.3,
    "max_tokens": 8192,
    "stream": false,
    "stop_token_ids": [151643],
    "frequency_penalty": 1.2
}'
```

更多相关参数可以查看vllm文档中的[SamplingParams](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)。



#### CLI Demo

直接启动命令行服务，支持单论或多轮问答。

```shell
python qwen-14b-chat-stream.py 
		--base_moddel your_qwen_model_path 
		--qlora_model your_qwen_qlora_model_path 
		--is_multi_turn 1
```
