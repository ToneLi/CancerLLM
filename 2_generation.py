from peft import PeftModel
import torch
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, LlamaTokenizer
import json


number = "1153"
checkpoint = f"checkpoint-{number}"
data_path = "Cancer_data"
input_test_file_name = data_path + "/Cancer_QA_data_test_instruction.json"
save_file_name = f"Cancer_QA_test_cancer_pre_train_mixtural_fine_tune_bio_mixture_7b_output_{number}.json"
lora_weights = f"cancer_biomixtural_1_7b_CancerQA_pre_train_mixtural/{checkpoint}"
base_model_path = "common-models/BioMistral-7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer = LlamaTokenizer.from_pretrained(lora_weights)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    lora_weights,
    torch_dtype=torch.float16
)

def make_inference(instruction, context=None):
    if context:
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
    else:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, max_length=1500).to("cuda:0")

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.9)
        results = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return results


if __name__ == "__main__":
    fw = open(save_file_name, "w", encoding="utf-8")
    i = 0

    with open(input_test_file_name, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = json.loads(line.strip())
            instruction = line["instruction"]
            sentence = line["context"]
            ground_truth = line["response"]
            predicted = make_inference(instruction, sentence)
            i += 1
            print(i)

            Dic_ = {
                "sentence": sentence,
                "ground_truth": ground_truth,
                "predicted": predicted
            }
            fw.write(json.dumps(Dic_))
            fw.write("\n")
            fw.flush()

"""
CUDA_VISIBLE_DEVICES=2 python generation.py
"""
