nohup deepspeed --include=localhost:0,1,2,3 ../SFT/train_zephyr_7b_qlora.py --deepspeed --deepspeed_config ../finetuner/ds.json > ../logs/zephyr_chinese_7b_instruct.log 2>&1 &