import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import Accelerator
from tqdm import tqdm
from .abstract_model import prm
from ..utils.log_utils import get_logger
from ..utils.model_utils import remove_step_prefix
import os

logger = get_logger(__name__)
final_model_dir = os.environ['FAST'] + "/rpisano1/checkpoints/final_model"

local_model_path = os.environ['WORK'] + "/rpisano1/models"
local_tokenizer_path = os.environ['WORK'] + "/rpisano1/models/tokenizer"


class Qwen2_5LoraPRM(prm):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-Math-PRM-7B",
        lora_path: str = final_model_dir,
        redundancy_threshold: float = 0.15,
        validity_threshold: float = 0.5,
    ) -> None:
        super().__init__(validity_threshold=validity_threshold, redundancy_threshold=redundancy_threshold)

        # tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_tokenizer_path,
            trust_remote_code=True,
            local_files_only=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True
        )
        # load LoRA weights
        self.model = base_model #PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.bfloat16)
        self.model.eval()

        # accelerator for multi-GPU / batch inferencing
        self.accelerator = Accelerator()

        # separator for reasoning steps
        self.step_separator = "\n\n"
        self.step_separator_token_id = self.tokenizer.encode(self.step_separator, add_special_tokens=False)[0]

        self.prompt = """You are a mathematical reasoning evaluator. Your task is to analyze mathematical problem-solving steps.

For each solution step, you need to evaluate its validity (0 to 1.0):
   * 1: Completely correct mathematical reasoning
   * 0.5: Partially correct with some mistakes
   * 0: Completely incorrect
   * Use any value in between to indicate varying degrees of correctness

Requirements:
- Evaluate each step independently
- Provide scores as floating-point numbers
- Maintain mathematical rigor in your evaluation
- Consider mathematical accuracy, logical coherence, and solution efficiency

Example output format:
[0.8, -0.5, 1.0]

You will be presented with a mathematical problem and its step-by-step solution. Please analyze each step and provide your evaluation in the specified format.
"""

    def getitem_function(self, meta_data, index):
        """
        Called in the dataloader to prepare each item for the model:
        ### PRMBench/mr_eval/evaluator.py ###
            dataset = BaseEvalDataset(
                load_data_function=load_data_function,
                getitem_function=self.model.getitem_function,
                ...
            )

        ### PRMBench/mr_eval/tasks/base_dataset/base_evaluation_dataset.py ###
            self.meta_data = self.load_data_function()
            ...
            def __getitem__(self, index):
                return self.getitem_function(self.meta_data,index)

        ### PRMBench/mr_eval/evaluator.py ###
            task_dict = get_task_functions(task_name)
            load_data_function, evaluate_function, task_config = task_dict["load_data_function"], task_dict["evaluate_function"], task_dict["task_config"]

        ### mr_eval/tasks/__init__.py ###
            def get_task_functions(task_name):
                function_list = ["load_data_function","evaluate_function","task_config"]
                module = __import__(f"mr_eval.tasks.{task_name}.task", fromlist=["*"])
                res_dict = {func:getattr(module, func) for func in function_list}

        ### mr_eval/tasks/prmtest_classified/task.py ###
            Questa funzione legge dei JSONL, elimina i duplicati e crea un sample di confronto “correct” con dati originali.
            correct_sample = dict(idx=correct_idx,question=item["original_question"],steps=item["original_process"],error_steps=[],classification="correct")
            Ogni sample ha un idx come <classe>_<id> e raccolto in meta_data.

        used in to prepare the dataset then used in respond(self, dataloader) defined below.
        """
        item = meta_data[index]
        data_idx = item["idx"]
        steps = item.get("steps", [])
        question = item["question"]


        combined_steps = ""
        for step in steps:
            #cleaned = remove_step_prefix(step)
            combined_steps += step + self.step_separator

        original_input_for_prm = f"# Question\n\n{question}\n\n# Solution\n\n{combined_steps}"
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": original_input_for_prm},
        ]
        conv_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer.encode(conv_str, return_tensors="pt").squeeze(0)

        return {"idx": data_idx, "input_ids": input_ids}


    def extract_step_level_rewards(self, texts):
        results = []
        for text in texts:

            # find teh array of step-level rewards
            start_idx = text.find("[")
            end_idx = text.find("]", start_idx)
            if start_idx == -1 or end_idx == -1:
                results.append([])
                continue
            step_level_rewards_str = text[start_idx:end_idx + 1]
            try:
                step_level_rewards = eval(step_level_rewards_str)
                if isinstance(step_level_rewards, list):
                    results.append(step_level_rewards)
                else:
                    results.append([])
            except Exception as e:
                logger.error(f"Error evaluating step-level rewards: {e}")
                results.append([])

        return results



    def respond(self, dataloader) -> None:
        """
        ### PRMBench/mr_eval/evaluator.py ###
            ...
            dataset = BaseEvalDataset(...)
            dataloader = DataLoader(dataset, ..)
            self.model.respond(dataloader)          ### OPERATIONS TO BE DONE IN PLACE ??
            res_log = dataset.evaluate()

        ### PRMBench/mr_eval/tasks/base_dataset/base_evaluation_dataset.py ###
            def store_results(self,result):
                self.results.append(result)

            def evaluate(self):
                return self.evaluate_function(results=self.results,meta_data=self.full_data)

        ### mr_eval/tasks/prmtest_classified/task.py ###

        """
        # prepare model and dataloader
        self.model, dataloader = self.accelerator.prepare(self.model, dataloader)
        self.accelerator.wait_for_everyone()
        self.model.eval()

        progress = tqdm(total=len(dataloader), desc="Model Responding")
        if len(dataloader) == 0:
            self.accelerator.wait_for_everyone()
            return

        with torch.no_grad():
            for batch in dataloader:
                idxs = batch["idx"]
                inputs = batch["input_ids"].to(self.accelerator.device)
                attention_mask = (inputs != self.tokenizer.pad_token_id).long().to(self.accelerator.device)
                attention_mask[inputs == self.step_separator_token_id] = 0

                # maybe set temperature to 0.1 for more stable
                generated_ids = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=False,
                )
                texts = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                rewards = self.extract_step_level_rewards(texts)

                for i, data_idx in enumerate(idxs):
                    try:
                        sl_validity = rewards[i]
                        score_dict = {
                            "step_level_validity_scores": sl_validity,
                            "step_level_validity_labels": [float(s) > self.validity_threshold for s in sl_validity],
                        }
                        res = {"scores": score_dict, "idx": data_idx}
                    except Exception as e:
                        print(f"Error processing idx {data_idx}: {e}")
                        res = {"scores": {}, "idx": data_idx, "validity": False}
                    dataloader.dataset.store_results(res)

                progress.update(1)

        self.accelerator.wait_for_everyone()
