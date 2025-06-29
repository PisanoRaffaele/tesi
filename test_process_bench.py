from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from peft import PeftModel
import os
import re
import json
import numpy as np
from vllm import LLM, SamplingParams
from collections import Counter
import time
import torch.distributed as dist

# tokenizer = AutoTokenizer.from_pretrained(
#     local_model_path,
#     trust_remote_code=True,
#     local_files_only=True
# )

def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None

def prepare_input_boxed(input_d):
    problem = input_d['problem']
    steps = input_d['steps']
    tagged_response = ''
    for sdx, step in enumerate(steps):
        tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    tagged_response = tagged_response.strip()
    prompt = f"""The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

"""+r"""Please put your final answer (i.e., the index) in \boxed{{}}."""
    messages = [{'role': 'user', 'content': prompt}]
    return messages


def apply_chat_template(toker, messages):
    input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    #print(f"Input prompt: {input_prompt}")
    return toker(input_prompt, add_special_tokens=False).input_ids

def main():

	save_model_dir = os.environ['SCRATCH'] + "/model_ft_2_nl"
	final_model_dir = os.environ['FAST'] + "/rpisano1/tokenized_dataset_2_chat_no_lables/final_model"
	local_model_path = os.environ['WORK'] + "/rpisano1/models"

	base_model = AutoModelForCausalLM.from_pretrained(
		local_model_path,
		trust_remote_code=True,
		#torch_dtype=torch.float16,
		#device_map="auto",
		local_files_only=True
	)

	model_ft = PeftModel.from_pretrained(
		base_model,
		final_model_dir,
		local_files_only=True
	)
	model_ft = model_ft.merge_and_unload()

	model_ft.save_pretrained(save_model_dir)

	print(type(model_ft))
	return

	rank = dist.get_rank()
	world_size = dist.get_world_size()

	print(f"Rank: {rank}, World Size: {world_size}")

	data_dir = os.environ['FAST'] + "/rpisano1/dataset/ProcessBench"

	output_dir = os.environ['FAST'] + "/rpisano1/ProcessBench/outputs/ft2_no_labels"

	#final_model_dir = os.environ['FAST'] + "/rpisano1/checkpoints/final_model"

	use_voting = False  #Set to False to disable voting

	llm = LLM(
			model=save_model_dir, tokenizer=local_model_path,
			gpu_memory_utilization=0.95,
			tensor_parallel_size=world_size,
			enable_prefix_caching=True, swap_space=0,
			max_num_seqs=32,
		)
	tokenizer = AutoTokenizer.from_pretrained(
		local_model_path,
		trust_remote_code=True,
		local_files_only=True
	)

	if use_voting:
		sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, n=8, max_tokens=8192, seed=42)
	else:
		sampling_params = SamplingParams(temperature=0., max_tokens=8192, seed=42)


	configs=['gsm8k', 'math', 'olympiadbench', 'omnimath']

	for config in configs:
		print(f"Processing {config}...")
		input_data = load_dataset('Qwen/ProcessBench', split=config, cache_dir=data_dir)
		# input_data = input_data.shard(num_shards=num_shards, index=shard_id)
		# output_dir_in = os.path.join(output_dir, config)

		prompt_token_ids = [apply_chat_template(tokenizer, prepare_input_boxed(e)) for e in input_data]

		generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
		res_data = []
		for i in range(len(input_data)):
			d = input_data[i].copy()

			if not use_voting:
				generated_critique = generations[i].outputs[0].text
				pred = extract_answer(generated_critique)
				try:
					pred = int(pred)
				except:
					pred = None
			else:
				generated_critique = [ee.text for ee in generations[i].outputs]
				preds = [extract_answer(e) for e in generated_critique]
				preds = [e for e in preds if e is not None]
				if len(preds) == 0:
					pred = None
				else:
					pred = Counter(preds).most_common(1)[0][0]
					try:
						pred = int(pred)
					except:
						pred = None

			d['generated_critique'] = generated_critique
			d['prediction'] = pred
			d['match'] = (pred == d['label'])

			res_data.append(d)

		error_data = [e for e in res_data if e['label'] != -1]
		correct_data = [e for e in res_data if e['label'] == -1]
		with open(os.path.join(output_dir, f'{config}_error.jsonl'), 'w') as f:
			for e in error_data:
				f.write(json.dumps(e) + '\n')
		with open(os.path.join(output_dir, f'{config}_correct.jsonl'), 'w') as f:
			for e in correct_data:
				f.write(json.dumps(e) + '\n')

		acc1 = np.mean([e['match'] for e in error_data]) * 100
		acc2 = np.mean([e['match'] for e in correct_data]) * 100
		f1 = 2 * acc1 * acc2 / (acc1 + acc2)
		print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    # Imposta la GPU locale su cui lavorare
    torch.cuda.set_device(int(dist.get_rank()))
    main()
