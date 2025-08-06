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
	reasoning_steps = reasoning_steps.replace("Step", "\nки\nStep").replace("\nки\nStep 1", "Step 1")
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
	
	# for debugging purposes
	#print(f"Prompt:\n{prompt}\n")

	return prompt


def main():

	lora_final_model_dir1 = os.environ['FAST'] + "/rpisano1/checkpoints_4_NEW_PRM800k_PDDL/"
	lora_final_model_dir2 = os.environ['FAST'] + "/rpisano1/checkpoints_3_NEW_PRM800k/"

	base_model1 = AutoModelForCausalLM.from_pretrained(
		local_model_path,
		trust_remote_code=True,
		torch_dtype=torch.float16,
		local_files_only=True,
	)

	base_model2 = AutoModelForCausalLM.from_pretrained(
		local_model_path,
		trust_remote_code=True,
		torch_dtype=torch.float16,
		local_files_only=True,
	)

	reward_model1 = RewardModel(base_model1)
	reward_model2 = RewardModel(base_model2)

	reward_model1 = PeftModel.from_pretrained(
		reward_model1,
		lora_final_model_dir1,
		torch_dtype=torch.float16
	)

	reward_head_path1 = os.path.join(lora_final_model_dir1, "reward_head.pt")
	reward_model1.reward_head.load_state_dict(torch.load(reward_head_path1, map_location="cuda"))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	reward_model1.to(device)
	reward_model1.eval()

	reward_model2 = PeftModel.from_pretrained(
		reward_model2,
		lora_final_model_dir2,
		torch_dtype=torch.float16
	)

	reward_head_path2 = os.path.join(lora_final_model_dir2, "reward_head.pt")
	reward_model2.reward_head.load_state_dict(torch.load(reward_head_path2, map_location="cuda"))
	reward_model2.to(device)
	reward_model2.eval()

	world_size = dist.get_world_size()

	llm = LLM(
		model=local_model_path, tokenizer=local_model_path,
		gpu_memory_utilization=0.35,
		tensor_parallel_size=world_size,
		enable_prefix_caching=True, swap_space=0,
		max_num_seqs=32,
	)

	sampling_params = SamplingParams(temperature=0.7, max_tokens=4096)

	def apply_chat_template(toker, messages):
		input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
		return toker(input_prompt, add_special_tokens=False).input_ids
	
	def extract_answer(solution_text: str):
		boxed_pattern = r'\\boxed\{([^}]*)\}'
		matches = re.findall(boxed_pattern, solution_text)
		if matches:
			return matches[-1].strip()
		return None

	def generate_n_candidates(prompt_token_ids):
		# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
		# outputs = base_model.generate(
		# 	input_ids,
		# 	do_sample=True,
		# 	temperature=0.8,
		# 	top_p=0.95,
		# 	num_return_sequences=n,
		# 	max_new_tokens=max_new_tokens,
		# )
		# return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
		generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
		generated_critiques = [gen.outputs[0].text for gen in generations]
		answers = [extract_answer(critique) for critique in generated_critiques]
		for i, answer in enumerate(answers):
			if answer is None:
				answers[i] = "NOT FOUND"

		return generated_critiques, answers

	def call_PRM(problem, batch, answers):
		prompts = [prepare_input_boxed(problem, ex) for ex in batch]
		batch_tokenized = tokenizer(
			prompts,
			return_tensors="pt",
			padding=True,
			truncation=True,
		)
		batch_tokenized = {k: v.to(device) for k, v in batch_tokenized.items()}
		for i in range(len(batch)):
			is_marker = torch.isin(batch_tokenized['input_ids'][i], torch.tensor(token_marker, device=batch_tokenized['input_ids'][i].device))
			batch_tokenized['attention_mask'][i][is_marker] = 0

		with torch.inference_mode():
			out1 = reward_model1(
				input_ids=batch_tokenized["input_ids"],
				attention_mask=batch_tokenized["attention_mask"],
			)

			out2 = reward_model2(
				input_ids=batch_tokenized["input_ids"],
				attention_mask=batch_tokenized["attention_mask"],
			)

		reward_logits1 = out1["reward_logits"].detach().cpu()
		if torch.isnan(reward_logits1).any():
			print("⚠️ Warning: NaN values found in logits!")
		del out1

		reward_logits2 = out2["reward_logits"].detach().cpu()
		if torch.isnan(reward_logits2).any():
			print("⚠️ Warning: NaN values found in logits!")
		del out2
		

		marker_id = tokenizer.encode(f"ки", add_special_tokens=False)
		input_ids = batch_tokenized["input_ids"].detach().cpu()
		del batch_tokenized["input_ids"], batch_tokenized["attention_mask"]
		del batch_tokenized
		if torch.isnan(input_ids).any():
			print("⚠️ Warning: NaN values found in input_ids!")
		marker_tensor = torch.tensor(marker_id)
		is_marker = torch.isin(input_ids, marker_tensor)

		# need to choose the highest scored reasoning
		max_scores_product_1 = 0
		max_score_idx_1 = -1
		for k in range(reward_logits1.size(0)):
			scores_i = []
			idxs = is_marker[k].nonzero(as_tuple=True)[0]
			if len(idxs) == 0:
				raise ValueError(f"⚠️ Nessun marker 'ки' trovato nel sample {k}")

			logits_i = reward_logits1[k, idxs]
			for logits in logits_i:
				prob = logits.softmax(dim=-1)[1].item()
				scores_i.append(prob)

			score_product = torch.tensor(scores_i).prod().item()
			max_scores_product_1 = max(max_scores_product_1, score_product)
			if score_product == max_scores_product_1:
				max_score_idx_1 = k

		if max_score_idx_1 == -1:
			raise ValueError("⚠️ Nessun indice massimo trovato per il punteggio")
		
		max_scores_product_2 = 0
		max_score_idx_2 = -1
		for k in range(reward_logits2.size(0)):
			scores_i = []
			idxs = is_marker[k].nonzero(as_tuple=True)[0]
			if len(idxs) == 0:
				raise ValueError(f"⚠️ Nessun marker 'ки' trovato nel sample {k}")

			logits_i = reward_logits2[k, idxs]
			for logits in logits_i:
				prob = logits.softmax(dim=-1)[1].item()
				scores_i.append(prob)

			score_product = torch.tensor(scores_i).prod().item()
			max_scores_product_2 = max(max_scores_product_2, score_product)
			if score_product == max_scores_product_2:
				max_score_idx_2 = k

		if max_score_idx_2 == -1:
			raise ValueError("⚠️ Nessun indice massimo trovato per il punteggio 2")

		gc.collect()
		torch.cuda.empty_cache()
		return (
			[batch[max_score_idx_1], max_scores_product_1, answers[max_score_idx_1], answers],
			[batch[max_score_idx_2], max_scores_product_2, answers[max_score_idx_2], answers]
		)

	def best_of_n(problems, n=8):
		prompt_token_ids = []
		for problem in problems:
			prompt = f"Solve this problem: {problem}\n Please reason step by step, start each step with 'Step i:' and put your final answer within \\boxed{{}}."
			messages = [{'role': 'user', 'content': prompt}]
			prompt_token_ids.extend([apply_chat_template(tokenizer, messages)] * n)

		candidates, answers = generate_n_candidates(prompt_token_ids)
		result_list = []
		for i, problem in enumerate(problems):
			candidates_i = candidates[i * n:(i + 1) * n]
			answers_i = answers[i * n:(i + 1) * n]
			res_i = call_PRM(problem, candidates_i, answers_i) #best_candidate, best_score, best_answer, answers
			result_list.append(res_i)
		return result_list, candidates

	def check_solution_matching(proposal, answer):
		# prompt = f"Your task is to determine if the proposed answer matches the solution to the problem.\n\nProposed answer: {proposal}\n\nReal answer: {answer}\n\nDo they match? Answer ONLY with 'yes' or 'no'. Response:"
		# messages = [{'role': 'user', 'content': prompt}]
		# prompt_token_ids = [apply_chat_template(tokenizer, messages)] * 5
		# generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=SamplingParams(temperature=0., max_tokens=1024, seed=42))

		# num_yes = sum('yes' in generation.outputs[0].text.strip().lower() for generation in generations)
		# num_no = sum('no' in generation.outputs[0].text.strip().lower() for generation in generations)

		# if num_yes > num_no:
		# 	return True
		# elif num_no > num_yes:
		# 	return False
		# else:
		# 	return False
		try:
			real_answer = answer.split("####")[-1].strip()
		except IndexError:
			print(f"⚠️ Errore nell'estrazione della risposta reale da: {answer}")
			real_answer = "NOT FOUND"
		proposal = proposal.strip()
		if real_answer in proposal or proposal in real_answer:
			return True
		else:
			return False

	# def score_problem(problem, solution, n=8):
	# 	best_candidate, best_score, best_answer, answers = best_of_n(problem, n=n)
	# 	match = False
	# 	if check_solution_matching(best_answer, solution):
	# 		match = True
	# 	other_match = False
	# 	if not match:
	# 		for answer in answers:
	# 			if check_solution_matching(answer, solution):
	# 				other_match = True
	# 				break
	# 	return {
	# 		"problem": problem,
	# 		"best_candidate": best_candidate,
	# 		"best_score": best_score,
	# 		"best_answer": best_answer,
	# 		"answers": answers,
	# 		"solution": solution,
	# 		"match": match,
	# 		"other_match": other_match
	# 	}
	
	def score_problems(problems, solutions, n=8):
		#best_candidate, best_score, best_answer, answers for each problem
		ret_list = []
		result_list, candidates = best_of_n(problems, n=n)
		for i in range(len(problems)):
			candidates_i = candidates[i * n:(i + 1) * n]
			# print(type(result_list[i]))
			# print(type(result_list))
			tuple = result_list[i] # info1 : best_candidate, best_score, best_answer, answers
			info1 = tuple[0]
			info2 = tuple[1]
			match_1 = False
			if check_solution_matching(info1[2], solutions[i]):
				match_1 = True

			match_2 = False
			if check_solution_matching(info2[2], solutions[i]):
				match_2 = True

			if match_1 or match_2:
				best_match = True
			else:
				best_match = False
				for answer in info1[3]:
					if check_solution_matching(answer, solutions[i]):
						best_match = True
						break
			ret_list.append({
				"problem": problems[i],
				"solution": solutions[i],
				"answers": info1[3],
				"candidates": candidates_i,

				"best_candidate_1": info1[0],
				"best_score_1": info1[1],
				"best_answer_1": info1[2],
				"best_candidate_2": info2[0],
				"best_score_2": info2[1],
				"best_answer_2": info2[2],

				"match_1": match_1,
				"match_2": match_2,
				"Best_match": best_match,
			})

		return ret_list

	def process_dataset(dataset, ds_name, n=8):
		results = []
		batch_size = 8
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
				print(f"— Salvataggio checkpoint: batch {batch_idx+1}/{total_batches}")
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
		total_matched_1 = sum(1 for r in results if r['match_1'])
		total_matched_2 = sum(1 for r in results if r['match_2'])
		total_best_matched = sum(1 for r in results if r['Best_match'])

		print(f"Total problems: {total_problems}, Matched 1: {total_matched_1}, Matched 2: {total_matched_2}, Best matched: {total_best_matched}")
		print(f"Accuracy 1: {total_matched_1 / total_problems * 100:.2f}%")
		print(f"Accuracy 2: {total_matched_2 / total_problems * 100:.2f}%")
		print(f"Best match accuracy: {total_best_matched / total_problems * 100:.2f}%")

	datasets_dir = os.environ['FAST'] + "/rpisano1/datasets/"

	ds_local = load_dataset("json", data_files={
		"train": datasets_dir + "gsm8k_train.json",
		"test": datasets_dir + "gsm8k_test.json"
	})

	process_dataset(ds_local['test'], "gsm8k_test", n=8)

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    # Imposta la GPU locale su cui lavorare
    torch.cuda.set_device(int(dist.get_rank()))
    main()
