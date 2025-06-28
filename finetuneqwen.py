from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
import os

print("🚀 Inizio script...")

data_dir = os.environ['FAST'] + "/rpisano1/dataset/"
checkpoint_dir = os.environ['FAST'] + "/rpisano1/checkpoints_2_2/"

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

# 	new_input = """You are a mathematical reasoning evaluator. Your task is to analyze mathematical problem-solving steps.

# For each solution step, you need to evaluate its validity (0 or 1):
#    * 1: Correct mathematical reasoning
#    * 0: Incorrect
# Example output format for 3 steps:
# [1, 1, 0]

# Requirements:
# - Evaluate each step independently
# - Maintain mathematical rigor in your evaluation
# - Consider mathematical accuracy, logical coherence, and solution efficiency

# Please analyze each step and provide your evaluation in a list at the end.
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

#     if example['info'] == "Math-Shepherd":
#         marker = ' киRESPки'
#         mapping = {1.0: '1', 0.0: '0'}
#         label_nums = [mapping.get(lbl) for lbl in example['label']]
#         if None in label_nums:
#             raise ValueError(f"Label non valida in {example['label']}")
#         label_string = "[" + ", ".join(label_nums) + "]"

#         label_text = f"{label_string}".strip()
#         in_te = process_math_shepherd_prompt(example['input'])
#         #input_text_1 = f"{example['input']}{marker}".strip()
#         input_text = f"{in_te}{marker}".strip()
#         #print(f"Input text: {input_text}")

#         #input_ids_1 = tokenizer_1.encode(input_text_1, add_special_tokens=False)
#         #input_ids_2 = tokenizer_1.encode(input_text, add_special_tokens=False)
#         #print(len(input_ids_1))
#         #print(len(input_ids_2))
#         #print(f"Input text original: {example['input']}")
#         #print(f"Input text: {input_text}")
#     else:
#         marker = ' киRESPки'
#         label_text = f"{example['Rating']}".strip()
#         input_text = f"{example['input']}{marker}".strip()

#     def find_subsequence(lst, sub):
#         len1, len2 = len(lst), len(sub)
#         indices = []
#         for i in range(len1 - len2 + 1):
#             if lst[i:i+len2] == sub:
#                 indices.append(i)
#         return indices

#     tokenized_inputs = tokenizer(
#         input_text,
#         truncation=True,
#         padding='max_length',
#         max_length=830,
#     )
#     length = len(tokenized_inputs['input_ids'])
#     tokenized_inputs['labels'] = [-100] * length

#     marker_ids = tokenizer.encode(f"{marker}", add_special_tokens=False)

#     indices = find_subsequence(tokenized_inputs['input_ids'], marker_ids)

#     if len(indices) == 0:
#         print(tokenized_inputs['input_ids'][-(100):])
#         print(f"Marcatore_ids: {marker_ids}")
#         print(f"Marcatore: {marker}")
#         print(f"input_text: {input_text}")
#         raise ValueError("Marcatore non trovato nell'input_ids")

#     label_ids = tokenizer.encode(label_text, add_special_tokens=False)

#     label_test_check = tokenizer.decode(label_ids, skip_special_tokens=False)

#     if label_test_check != label_text:
#         print(f"label_text: {label_text}")
#         print(f"label_test_check: {label_test_check}")
#         raise ValueError("Label text non corrisponde alla decodifica")

#     start_idx = indices[0]

#     if start_idx + len(label_ids) > 830:
#         limit = 830 - start_idx
#         if example['info'] == "Math-Shepherd":
#             label_ids = label_ids[:limit]
#             print(f"Truncated label_ids to match limit length: {len(label_ids)}")
#         else:
#             raise ValueError("Lunghezza del marcatore non corrisponde alla lunghezza dell'etichetta")
#     end_idx = start_idx + len(label_ids)
#     end_marker_idx = start_idx + len(marker_ids)

#     if end_idx > length:
#         raise ValueError("Lunghezza dell'etichetta supera la lunghezza dell'input_ids")

#     tokenized_inputs['labels'][start_idx:end_idx] = label_ids
#     tokenized_inputs['attention_mask'][start_idx:end_marker_idx] = [0] * len(marker_ids)
#     #print(tokenized_inputs['labels'])
#     #print(tokenized_inputs['input_ids'])
#     #print(tokenized_inputs['attention_mask'])

#     #print(f"--> start_idx: {start_idx}, end_idx: {end_idx}")
#     #print("marker_ids:", marker_ids)
#     #print("marker:", marker)
#     #print("label_ids:", label_ids)
#     #print("label_text:", label_text)
#     #print("sequence:", tokenized_inputs['input_ids'])
#     #print(f"tokenized_inputs['labels']: {tokenized_inputs['labels']}")
#     #print("found at:", indices)

#     assert len(marker_ids) > 0, "marker_ids vuoto!"
#     assert start_idx + len(label_ids) <= len(tokenized_inputs['input_ids']), f"label out of bounds: {start_idx} + {len(label_ids)} > {len(tokenized_inputs['input_ids'])}"
#     assert any(l != -100 for l in tokenized_inputs['labels']), "Non sto effettivamente supervisionando NESSUN token!"
#     assert len(tokenized_inputs['input_ids']) == len(tokenized_inputs['labels']), "input_ids e labels non hanno la stessa lunghezza!"
#     assert len(tokenized_inputs['input_ids']) == len(tokenized_inputs['attention_mask']), "input_ids e attention_mask non hanno la stessa lunghezza!"
#     return tokenized_inputs

# print("📚 Caricamento dataset...")


# dataset = load_dataset("json", data_files=os.path.join(data_dir, json_file), split="train")
# dataset = dataset.filter(
#     lambda x: not (x["info"] == "Math-Shepherd" and len(x["label"]) > 22)
# )

# tokenized_dataset = dataset.train_test_split(test_size=0.1)
# tokenized_dataset = tokenized_dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=32)

# print(tokenized_dataset["train"].column_names)

# tokenized_dataset.save_to_disk(f"{data_dir}/tokenized_dataset_2_2")

tokenized_dataset = load_from_disk(f"{data_dir}/tokenized_dataset_2_2")

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
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir=f"{checkpoint_dir}/logs",
    logging_steps=2000,
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

trainer.save_model(f"{checkpoint_dir}/final_model")


# candidate_tokens1 = tokenizer.encode(f" + ")
# candidate_tokens2 = tokenizer.encode(f" - ")
# def preprocess_function(example):

# 	if example['info'] == "Math-Shepherd":
# 		input_text = f"{example['input']}".strip()
# 		label_text = f"{example['label']}".strip()
# 		marker = ' ки'

# 	else:
# 		marker = ' ки.кики'
# 		label_text = f"{example['Rating']}".strip()
# 		input_text = f"{example['input']} {marker}".strip()


# 	def find_subsequence(lst, sub):
# 		len1, len2 = len(lst), len(sub)
# 		indices = []
# 		for i in range(len1 - len2 + 1):
# 			if lst[i:i+len2] == sub:
# 				indices.append(i)
# 		return indices

# 	tokenized_inputs = tokenizer(
# 		input_text,
# 		truncation=True,
# 		padding='max_length',
# 		max_length=810,
# 	)
# 	length = len(tokenized_inputs['input_ids'])
# 	tokenized_inputs['labels'] = [-100] * length

# 	marker_ids = tokenizer.encode(f"{marker}", add_special_tokens=False)

# 	indices = find_subsequence(tokenized_inputs['input_ids'], marker_ids)

# 	if len(indices) == 0:
# 		print(tokenized_inputs['input_ids'][-(100):])
# 		print(f"Marcatore_ids: {marker_ids}")
# 		print(f"Marcatore: {marker}")
# 		print(f"input_text: {input_text}")
# 		raise ValueError("Marcatore non trovato nell'input_ids")

# 	if example['info'] == "Math-Shepherd":
# 		for i, start_idx in enumerate(indices):
# 			end_idx = start_idx + len(marker_ids)
# 			assert len(candidate_tokens1) == len(marker_ids), "no equal len"
# 			if example['label'][i] == '+':
# 				tokenized_inputs['labels'][start_idx: end_idx] = candidate_tokens1
# 			elif example['label'][i] == '-':
# 				tokenized_inputs['labels'][start_idx: end_idx] = candidate_tokens2
# 			else:
# 				raise ValueError('label is wrong')
# 			tokenized_inputs['attention_mask'][start_idx: end_idx] = [0] * len(marker_ids)
# 	else:
# 		start_idx = indices[0]

# 		label_ids = tokenizer.encode(label_text, add_special_tokens=False)

# 		if len(marker_ids) < len(label_ids):
# 			print(len(label_ids), len(marker_ids))
# 			raise ValueError("Lunghezza del marcatore non corrisponde alla lunghezza dell'etichetta")
# 		end_idx = start_idx + len(label_ids)

# 		if end_idx > length:
# 			raise ValueError("Lunghezza dell'etichetta supera la lunghezza dell'input_ids")

# 		tokenized_inputs['labels'][start_idx:end_idx] = label_ids
# 		tokenized_inputs['attention_mask'][start_idx:end_idx] = [0] * len(label_ids)

# 	return tokenized_inputs
