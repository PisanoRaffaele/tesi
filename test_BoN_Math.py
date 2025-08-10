from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from peft import PeftModel
import os
import re
import json
import numpy as np
from collections import Counter
import time
import gc

from vllm import LLM, SamplingParams
import torch.distributed as dist

local_model_path = os.environ['FAST'] + "/rpisano1/models/qwen_7b_math_instruct"
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True, local_files_only=True)

token_marker = tokenizer.encode("ки", add_special_tokens=False)
id_zero, id_one = tokenizer.convert_tokens_to_ids(["0", "1"])

class RewardModel(PreTrainedModel):
	def __init__(self, base_model):
		super().__init__(base_model.config)
		self.model = base_model
		self.reward_head = torch.nn.Linear(base_model.config.hidden_size, 2).to(torch.float16)

		torch.nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.02)
		torch.nn.init.zeros_(self.reward_head.bias)

	def forward(self, input_ids, attention_mask=None, **kwargs):
		kwargs.pop("output_hidden_states", None)
		kwargs.pop("return_dict", None)

		with torch.autocast(device_type="cuda", dtype=torch.float16):
			outputs = self.model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				output_hidden_states=True,
				return_dict=True,
				**kwargs,
			)
		hidden = outputs.hidden_states[-1]
		raw_logits = self.reward_head(hidden)

		marker_ids_tensor = torch.tensor(token_marker, device=input_ids.device)
		is_marker = torch.isin(input_ids, marker_ids_tensor)


		reward_logits = torch.zeros_like(raw_logits)
		reward_logits[is_marker] = raw_logits[is_marker]

		return {
			"logits": outputs.logits,
			"reward_logits": reward_logits,
			"hidden_states": outputs.hidden_states,
			"past_key_values": outputs.get("past_key_values", None)
		}

	def generate(self, *args, **kwargs):
		return self.model.generate(*args, **kwargs)

	def prepare_inputs_for_generation(self, *args, **kwargs):
		return self.model.prepare_inputs_for_generation(*args, **kwargs)

def prepare_input_boxed(problem, reasoning_steps):
	num_matches_steps = len(re.findall(r"Step \d+", reasoning_steps))

	if num_matches_steps > 0:
		def replace_step(match):
			step_text = match.group()
			step_num = int(match.group(1))
			if step_num != 1:
				return f"ки\n{step_text}"
			else:
				return step_text

		reasoning_steps = re.sub(r"Step (\d+)", replace_step, reasoning_steps)

	else:
		print("reasoning_steps: ", reasoning_steps)

	reasoning_steps += "\nки\n"

	prompt = f"""You are a mathematical reasoning evaluator. Your task is to analyze mathematical problem-solving steps.

For each solution step, you need to evaluate its validity (0 or 1):
	* 1: Correct mathematical reasoning
	* 0: Incorrect

- Evaluate each step independently
- Maintain mathematical rigor in your evaluation
- Consider mathematical accuracy, logical coherence, and solution efficiency

Output ONLY the list of validity ratings.
Here is the problem and its solution steps:
{problem}\n{reasoning_steps}"""

	return prompt


def main():

	world_size = dist.get_world_size()

	llm = LLM(
		model=local_model_path, tokenizer=local_model_path,
		gpu_memory_utilization=0.9,
		tensor_parallel_size=world_size,
		enable_prefix_caching=True, swap_space=0,
		max_num_seqs=32,
	)

	sampling_params = SamplingParams(temperature=0.7, max_tokens=8192)

	def apply_chat_template(toker, messages):
		input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
		return toker(input_prompt, add_special_tokens=False).input_ids

	def extract_answer(solution_text: str):
		boxed_pattern = r'\\boxed\{([^}]*)\}'
		matches = re.findall(boxed_pattern, solution_text)
		if matches:
			return matches[-1].strip()
		return None

	def best_of_n(problems, n=8):
		prompt_token_ids = []
		N = n * 3
		for problem in problems:
			prompt = f"""Solve this problem: {problem}
Reason step by step, and output exactly this format:
Step 1: ...
Step 2: ...
...
Put your final answer within \\boxed{{}}."""
			messages = [{'role': 'user', 'content': prompt}]
			prompt_token_ids.extend([apply_chat_template(tokenizer, messages)] * N)

		generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

		generated_critiques = []
		print(len(problems), len(prompt_token_ids), len(generations))
		for i, problem in enumerate(problems):
			generated_critiques_i = []
			generations_i = generations[i * N:(i + 1) * N]

			for gen in generations_i:
				num_matches_steps = len(re.findall(r"Step \d+", gen.outputs[0].text))
				answer = extract_answer(gen.outputs[0].text)
				if num_matches_steps > 0 and answer is not None and not gen.outputs[0].text in generated_critiques_i:
					generated_critiques_i.append(gen.outputs[0].text)
				else:
					print(f"Skipping generation {gen.outputs[0].text} for problem {problem} due to format issues.")
					print("\n\n#########################################################################################################\n\n")


			while len(generated_critiques_i) < n:
				generated_critiques_i.append("")

			generated_critiques_i = generated_critiques_i[:n]
			generated_critiques.extend(generated_critiques_i)

		answers = [extract_answer(critique) for critique in generated_critiques]
		for i, answer in enumerate(answers):
			if answer is None:
				answers[i] = "###ERR###"

		return generated_critiques, answers

	def check_solution_matching(proposal, answer):
		try:
			real_answer = answer.split("####")[-1].strip()
		except IndexError:
			raise ValueError(f"Answer format error: {answer}")
		proposal = proposal.strip()
		if real_answer in proposal or proposal in real_answer:
			return True
		else:
			return False

	def score_problems(problems, solutions, n=8):
		ret_list = []
		candidates, answers = best_of_n(problems, n=n)
		for i in range(len(problems)):
			candidates_i = candidates[i * n:(i + 1) * n]
			answers_i = answers[i * n:(i + 1) * n]
			best_match = False
			for answer in answers_i:
				if check_solution_matching(answer, solutions[i]):
					best_match = True
					break
			ret_list.append({
				"problem": problems[i],
				"solution": solutions[i],
				"answers": answers_i,
				"candidates": candidates_i,

				"best_candidate_1": None,
				"best_score_1": None,
				"best_answer_1": None,
				"best_candidate_2": None,
				"best_score_2": None,
				"best_answer_2": None,

				"match_1": None,
				"match_2": None,
				"Best_match": best_match,
			})

		return ret_list

	def process_dataset(dataset, ds_name, n=8):
		results = []
		batch_size = 32
		checkpoint_interval = 10
		output_file = os.path.join(os.environ['FAST'] + "/rpisano1/BoN", f"{ds_name}_results.jsonl")

		processed_batches = 0
		if os.path.isfile(output_file):
			with open(output_file, 'r') as f:
				for line in f:
					results.append(json.loads(line))
			processed_batches = len(results) // batch_size
			print(f"⚠️ Ripreso da checkpoint: {processed_batches} batch già processati ({len(results)} problemi).")

		total_batches = (len(dataset) + batch_size - 1) // batch_size

		for batch_idx in range(processed_batches, total_batches):
			start = batch_idx * batch_size
			end = start + batch_size
			batch = dataset[start:end]
			if batch_idx % 50 == 0:
				print(f"Processing batch {batch_idx // batch_size + 1} of {len(dataset) // batch_size + 1}...")
			problems = batch['question']
			solutions = batch['answer']

			results.extend(score_problems(problems, solutions, n=n))

			if (batch_idx + 1) % checkpoint_interval == 0 or (batch_idx + 1) == total_batches:
				print(f"— Salvataggio checkpoint: batch {batch_idx+1}/{total_batches}; dataset size: {len(dataset)}")
				with open(output_file, 'w') as f:
					for r in results:
						f.write(json.dumps(r) + '\n')
		# for i, data in enumerate(dataset):
		# 	problem = data['question']
		# 	solution = data['answer']
		# 	if not problem or not solution:
		# 		continue
		# 	result = score_problem(problem, solution, n=n)
		# 	results.append(result)
		# 	if i % 100 == 0:
		# 		print(f"Processed {i} problems...")
		total_problems = len(dataset)
		#total_matched_1 = sum(1 for r in results if r['match_1'])
		#total_matched_2 = sum(1 for r in results if r['match_2'])
		total_best_matched = sum(1 for r in results if r['Best_match'])

		print(f"Total problems: {total_problems},  Best matched: {total_best_matched}")
		#print(f"Accuracy 1: {total_matched_1 / total_problems * 100:.2f}%")
		#print(f"Accuracy 2: {total_matched_2 / total_problems * 100:.2f}%")
		print(f"Best match accuracy: {total_best_matched / total_problems * 100:.2f}%")

	datasets_dir = os.environ['FAST'] + "/rpisano1/datasets/"

	# ds_local = load_dataset("json", data_files={
	# 	"train": datasets_dir + "gsm8k_train.json",
	# 	"test": datasets_dir + "gsm8k_test.json"
	# })

	ds_local = load_dataset("json", data_files={
		"test": datasets_dir + "minervamath_test.json",
	})

	# select half
	ds_local['test'] = ds_local['test'].select(range(len(ds_local['test']) // 2))

	process_dataset(ds_local['test'], "minervamath_test", n=8)

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    # Imposta la GPU locale su cui lavorare
    torch.cuda.set_device(int(dist.get_rank()))
    main()
