from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
import os

print("🚀 Inizio script...")

data_dir = os.environ['FAST'] + "/rpisano1/dataset/"
checkpoint_dir = os.environ['FAST'] + "/rpisano1/tokenized_dataset_2_3/"

local_model_path = os.environ['WORK'] + "/rpisano1/models/"

json_file=f'{data_dir}/final_data_2.json'

print("🔄 Caricamento modello...")

tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    local_files_only=True
)

# def process_math_shepherd_prompt(input_text):
# 	original_input = """You are a mathematical reasoning evaluator. Your task is to analyze mathematical problem-solving steps.

# For each solution step, you need to evaluate its validity (0 to 1.0):
#    * 1: Completely correct mathematical reasoning
#    * 0.5: Partially correct with some mistakes
#    * 0: Completely incorrect
#    * Use any value in between to indicate varying degrees of correctness

# Requirements:
# - Evaluate each step independently
# - Provide scores as floating-point numbers
# - Maintain mathematical rigor in your evaluation
# - Consider mathematical accuracy, logical coherence, and solution efficiency

# Example output format for 3 steps:
# [0.7, 1.0, 0.2]

# You will be presented with a mathematical problem and its step-by-step solution. Please analyze each step and provide your evaluation in the specified format.
# Here is the mathematical problem and its solution steps:
# """

# 	new_input = r"""You are a mathematical reasoning evaluator. Your task is to analyze mathematical problem-solving steps.

# For each solution step, you need to evaluate its validity (0 or 1):
#    * 1: Correct mathematical reasoning
#    * 0: Incorrect

# Requirements:
# - Evaluate each step independently
# - Maintain mathematical rigor in your evaluation
# - Consider mathematical accuracy, logical coherence, and solution efficiency

# Output the list of validity ratings. Example output format for 3 steps: \boxed{[1, 1, 0]}
# Here is the mathematical problem and its solution steps:
# """
# 	if original_input not in input_text:
# 		raise ValueError(f"Original input not found in the text: {input_text}")
# 	if ' ки' not in input_text:
# 		raise ValueError(f"Marker ' ки' not found in the text: {input_text}")

# 	ret = input_text.replace(original_input, new_input)
# 	ret = ret.replace(" ки", "")
# 	return ret

# def preprocess_function(example):
# 	marker = ' киRESPки'
# 	if example['info'] == "Math-Shepherd":
# 		mapping = {1.0: '1', 0.0: '0'}
# 		label_nums = [mapping.get(lbl) for lbl in example['label']]
# 		if None in label_nums:
# 			raise ValueError(f"Label non valida in {example['label']}")
# 		label_string = "[" + ", ".join(label_nums) + "]"

# 		label_text = f"{label_string}".strip()
# 	else:
# 		label_text = f"{example['Rating']}".strip()

# 	input_text = f"{example['input']}{marker}".strip()

# 	tokenized_inputs = tokenizer(
# 		input_text,
# 		truncation=True,
# 		padding='max_length',
# 		max_length=830,
# 	)

# 	def find_subsequence(lst, sub):
# 		len1, len2 = len(lst), len(sub)
# 		indices = []
# 		for i in range(len1 - len2 + 1):
# 			if lst[i:i+len2] == sub:
# 				indices.append(i)
# 		return indices

# 	length = len(tokenized_inputs['input_ids'])
# 	tokenized_inputs['labels'] = [-100] * length

# 	tokenized_marker = tokenizer.encode(marker, add_special_tokens=False)
# 	tokenized_label = tokenizer.encode(label_text, add_special_tokens=False)
# 	indices = find_subsequence(tokenized_inputs['input_ids'], tokenized_marker)

# 	if len(indices) == 0:
# 		print(tokenized_inputs['input_ids'][-(100):])
# 		print(f"Label text: {label_text}")
# 		raise ValueError("Label text non trovato nell'input_ids")
# 	start_idx = indices[0]

# 	if start_idx + len(tokenized_label) > 830:
# 		limit = 830 - start_idx
# 		if example['info'] == "Math-Shepherd":
# 			tokenized_label = tokenized_label[:limit]
# 			print(f"Truncated label_ids to match limit length: {len(tokenized_label)}")
# 		else:
# 			raise ValueError("Lunghezza del marcatore non corrisponde alla lunghezza dell'etichetta")
# 	end_idx = start_idx + len(tokenized_label)

# 	tokenized_inputs['labels'][start_idx:end_idx] = tokenized_label

# 	tokenized_inputs['attention_mask'][start_idx:] = [0] * (length - start_idx)

# 	return tokenized_inputs

# print("📚 Caricamento dataset...")

# dataset = load_dataset("json", data_files=os.path.join(data_dir, json_file), split="train")
# dataset = dataset.filter(
#     lambda x: not (x["info"] == "Math-Shepherd" and len(x["label"]) > 22)
# )

# dataset = dataset.shuffle(seed=42)

# tokenized_dataset = dataset.train_test_split(test_size=0.1, seed=42)
# tokenized_dataset = tokenized_dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=32)

# print(tokenized_dataset["train"].column_names)

# tokenized_dataset.save_to_disk(f"{data_dir}/tokenized_dataset_2_3")

tokenized_dataset = load_from_disk(f"{data_dir}/tokenized_dataset_2_3")

print(checkpoint_dir)

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
    num_train_epochs=2, #2
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50, #50
    weight_decay=0.01,
    logging_dir=f"{checkpoint_dir}/logs",
    logging_steps=4000, #2000
    eval_strategy="steps",
    save_strategy="epoch",
    save_steps=4000,
    learning_rate=3e-5,
    fp16=True,
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
print(checkpoint_dir)

trainer.save_model(f"{checkpoint_dir}/final_model")
