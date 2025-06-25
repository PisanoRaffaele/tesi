from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
import os

print("🚀 Inizio script...")

data_dir = os.environ['FAST'] + "/rpisano1/dataset/"
checkpoint_dir = os.environ['FAST'] + "/rpisano1/checkpoints_ftPRM/"

local_model_path = os.environ['FAST'] + "/rpisano1/qwenPRMbase"

# json_file=f'{data_dir}/final_data_3.json'

print("🔄 Caricamento modello...")

tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    local_files_only=True
)

# def preprocess_function(example):
#     if example['info'] == "Math-Shepherd":
#         return None
#         lenn = len(example['label'])
#         marker = " [" + ", ".join(["ки.ки"] * lenn) + "]"
#         label_text = f"{example['label']}"
#     elif example['info'] == "PRM-1w":
#         return None
#         marker = 'ки.'
#         label_text = f"{example['first_wrong']}"
#     else:
#         marker = 'ки.кики'
#         label_text = f"{example['Rating']}"

#     def find_subsequence(lst, sub):
#         len1, len2 = len(lst), len(sub)
#         indices = []
#         for i in range(len1 - len2 + 1):
#             if lst[i:i+len2] == sub:
#                 indices.append(i)
#         return indices

#     input_text = f"{example['input']} {marker}"
#     tokenized_inputs = tokenizer(
#         input_text,
#         truncation=True,
#         padding='max_length',
#         max_length=806,
#     )
#     length = len(tokenized_inputs['input_ids'])
#     tokenized_inputs['labels'] = [-100] * length

#     marker_ids = tokenizer.encode(f" {marker}", add_special_tokens=False)
#     indices = find_subsequence(tokenized_inputs['input_ids'], marker_ids)

#     if len(indices) == 0:
#         print(tokenized_inputs['input_ids'][-(100):])
#         print(f"Marcatore_ids: {marker_ids}")
#         if len(marker_ids) > 85 and example['info'] == "Math-Shepherd":
#             #skip this example if marker is too long
#             print(f"Skipping example with long marker: {marker}")
#             return None
#         raise ValueError("Marcatore non trovato nell'input_ids")
#     start_idx = indices[0]

#     label_ids = tokenizer.encode(label_text, add_special_tokens=False)

#     if len(marker_ids) < len(label_ids):
#         print(len(label_ids), len(marker_ids))
#         raise ValueError("Lunghezza del marcatore non corrisponde alla lunghezza dell'etichetta")
#     end_idx = start_idx + len(label_ids)

#     if end_idx > length:
#         raise ValueError("Lunghezza dell'etichetta supera la lunghezza dell'input_ids")

#     tokenized_inputs['labels'][start_idx:end_idx] = label_ids
#     tokenized_inputs['attention_mask'][start_idx:end_idx] = [0] * len(label_ids)

#     return tokenized_inputs

# print("📚 Caricamento dataset...")

# dataset = load_dataset("json", data_files=os.path.join(data_dir, json_file), split="train")

# tokenized_dataset = dataset.train_test_split(test_size=0.1)
# tokenized_dataset = tokenized_dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=32)

# tokenized_dataset.save_to_disk(f"{data_dir}/tokenized_dataset_only_pddl")

tokenized_dataset = load_from_disk(f"{data_dir}/tokenized_dataset_only_pddl")

print("⚙️ Configurazione PEFT/LoRA...")

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

num_gpus = torch.cuda.device_count()
print(f"🖥️ Numero GPU disponibili: {num_gpus}")

total_batch_size = 16

training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir=f"{checkpoint_dir}/logs",
    logging_steps=1500,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=4000,
    learning_rate=3e-5,
    #fp16=True,
    report_to='none',
    label_names=["labels"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

print("🏁 Inizio training...")
trainer.train()
print("✅ Fine training.")


trainer.save_model(f"{checkpoint_dir}/final_model")
