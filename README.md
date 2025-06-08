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

###   Instruction tuning
####  Data format 
##### Synthetic example  for cancer  diagnosis generation:
```
{
  "instruction": "You are a medical expert. This task involves generating the diagnosis based on the provided context or text.",
  "context": "1) Reason for visit: The patient reports a persistent cough and unintended weight loss over the past two months. 2) Treatment site: Right upper lobe of the lung. 3) Subjective information: The patient complains of shortness of breath, chest discomfort, and fatigue. 4) Nursing Review of Systems (ROS): Positive for night sweats, mild fever, and decreased appetite. 5) Objective observations: Decreased breath sounds and dullness to percussion on the right lung. 6) Laboratory test results: Chest X-ray reveals a mass in the right upper lobe; biopsy confirms malignant cells.",
  "response": "lung cancer"
}
```
##### Synthetic example for cancer phenotypes extraction:
```
{
  "instruction": "You are an excellent linguist. The task is to answer the question by given the context or text",
  "context": "In the text: The specimen shows a 2.5 cm firm white lesion located in the upper outer quadrant of the breast. What is the tumor size in the given context?",
  "response": "2.5",
}
```


####  How to fine-tune and generate  the answer:

Run the following code to train the model and generate the answers:

```
"""
CUDA_VISIBLE_DEVICES=9 nohup python cancerLLM_down_stream_fine_tuning.py > myout.cancerLLM_down_stream_fine_tuning 2>&1 &
CUDA_VISIBLE_DEVICES=2 python generation.py
"""
```

