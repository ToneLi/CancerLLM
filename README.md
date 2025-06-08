# CancerLLM

###  Introduction
In this work, we proposed CancerLLM, a model with 7 billion parameters and a Mistral-style architecture,
pre-trained on nearly 2.7M clinical notes and over 515K pathology reports covering
17 cancer types, followed by fine-tuning on two cancer-relevant tasks, including
cancer phenotypes extraction and cancer diagnosis generation.


###   Model Pre-training
We release the pre-training code to facilitate model pre-training for the specific domain.
Step 1: Collect the clinical notes and place them in the ./data directory.
Step 2: Run the following command:
```
CUDA_VISIBLE_DEVICES=5,6,8 torchrun --nproc_per_node 2 pretraining.py \
  --model_type auto \
  --model_name_or_path '/scratch/ahcie-gpu2/common-models/Mistral-7B-Instruct-v0.1' \
  --train_file_dir "./data" \
  --validation_file_dir "./data" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --do_train \
  --do_eval \
  --use_peft True \
  --seed 42 \
  --max_train_samples 4067420 \
  --max_eval_samples 1000 \
  --num_train_epochs 0.5 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 200000 \
  --evaluation_strategy steps \
  --save_steps 200000 \
  --save_strategy steps \
  --save_total_limit 13 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 10 \
  --block_size 3000 \
  --group_by_length True \
  --output_dir outputs-pt-qwen-v1 \
  --overwrite_output_dir \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --device_map auto \
  --report_to tensorboard \
  --ddp_find_unused_parameters False \
  --gradient_checkpointing True \
  --cache_dir ./cache

```
