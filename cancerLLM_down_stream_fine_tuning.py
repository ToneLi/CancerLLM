import torch
import transformers
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import LlamaTokenizer
from trl import SFTTrainer
from datasets import load_dataset

def formatting_func(example):
    if example.get("context", ""):
        input_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input:\n"
            f"{example['context']}\n\n"
            "### Response:\n"
            f"{example['response']}"
        )
    else:
        input_prompt = (
            "Below is an instruction that describes a task.\n\n"
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Response:\n"
            f"{example['response']}"
        )
    return {"text": input_prompt}

def prepare_data(path):
    data = load_dataset("json", data_files=path)
    formatted_data = data.map(formatting_func)
    return formatted_data["train"]

data_path = "Cancer_data"
train_path = data_path + "/Cancer_QA_data_train_instruction.json"
dev_path = data_path + "/Cancer_QA_data_dev_instruction.json"

train = prepare_data(train_path)
dev = prepare_data(dev_path)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_weights = "0_Pre_train_LLM/outputs_cancer_LLM/checkpoint-28779"

model_id = "common-models/Mistral-7B-Instruct-v0.1"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
base_model.enable_input_require_grads()

base_model = PeftModel.from_pretrained(
    base_model,
    lora_weights,
    torch_dtype=torch.float16,
    is_trainable=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=train,
    eval_dataset=dev,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        logging_steps=1,
        learning_rate=2e-4,
        num_train_epochs=5,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        output_dir="cancer_biomixtural_1_7b_CancerQA_pre_train_mixtural",
        optim="paged_adamw_8bit",
        fp16=True,
        evaluation_strategy="epoch",
        eval_steps=0.2,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
    ),
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=4000
)

supervised_finetuning_trainer.train()

# CUDA_VISIBLE_DEVICES=9 nohup python 0_biomixtural_7b_single_fine_tuning.py > myout.0_biomixtural_7b_single_fine_tuning 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python 0_llama3_8b_single_fine_tuning.py > myout.0_llama3_8b_single_fine_tuning 2>&1 &
