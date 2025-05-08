from typing import List
from abc import ABC, abstractmethod
import torch
# from gemma import gm
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

SUBTASK_LIST = [
    "close [object]: for example, close the door",
    "screw [object]: for example, screw the light bulb", 
    "open [object]: for example, open the door",
    "pick up [object] [and lift it up to [target]]: for example, pick up the cup and lift it up to the shelf, pick up the block",
    "put [object] [in|on] [target]: for example, put the cup on the shelf, put the block in the box on the table",
    "push [object]: for example, push the red button",
    "use the [tool] to drag [object] to [target]: for example, use the stick to drag the cup to the shelf",
    "slide [object] to [target]: for example, slide the block to the target",
    "stack [objects]: for example, stack 2 navy cups, stack orange block on the blue block",
]
PLANNING_PROMPT = """Split the instruction into few subtasks, use '|' to split it.
The subtask should in the list:\n""" + \
'\n'.join([f"{i+1}. {subtask}" for i, subtask in enumerate(SUBTASK_LIST)]) + \
"""
Example:
[Instruction] put the cup on the shelf, then push the red button
[Answer] take the cup|put on the shelf|push the red button

[Instruction] {instruction}
[Answer] 
"""

VALIDATION_PROMPT = """
Is subgoal {subgoal} is achieved in the given current scenario.
Only answer 'yes' or 'no'."""

class PlanAgent(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_plan(self, instruction_list: List[str]) -> List[str]:
        return instruction_list[0].split(',')
    
    @abstractmethod
    def verify_subgoal(self, subgoal: str, current_observation_path) -> tuple:
        return True, None
    
class GemmaPlanAgent(PlanAgent):
    def __init__(self):
        super().__init__()
        model_id = "google/gemma-3-12b-it"
        self.processor = AutoProcessor.from_pretrained(model_id)
        # self.model = AutoModelForImageTextToText.from_pretrained(model_id)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
        # self.pipe = pipeline("image-text-to-text", model=model_id)
        self.planning_prompt = PLANNING_PROMPT
        
        self.validation_prompt = VALIDATION_PROMPT
    
    def get_plan(self, instruction_list: List[str]) -> List[str]:
        messages = [
            {
                "role": "user", "content": [
                    {"type": "text", "text": self.planning_prompt.format(instruction=instruction_list[0])}
                ]
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

            output = self.processor.decode(generation, skip_special_tokens=True)
        answer_string = '[Answer] '
        if answer_string in output:
            output = output[output.rfind(answer_string)+len(answer_string):]
        # output = self.pipe(self.prompt.format(instruction=self.instruction_list[0]))[0]['generated_text']
        # print(output)
        return output.split('|')
    
    def verify_subgoal(self, subgoal: str, current_observation_path: str) -> (bool, str):
        messages = [
            {
                "role": "user", "content": [
                    {"type": "text", "text": self.validation_prompt.format(subgoal=subgoal)},
                    {"type": "image", "image": current_observation_path}
                ]
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )

        input_len = inputs["input_ids"].shape[-1]
        
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

            output = self.processor.decode(generation, skip_special_tokens=True)
        return 'yes' in output, output

class MistralPlanAgent(PlanAgent):
    def __init__(self):
        super().__init__()
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.planning_prompt = PLANNING_PROMPT
        self.validation_prompt = VALIDATION_PROMPT
        
    def get_plan(self, instruction_list: List[str]) -> List[str]:
        messages = [
            {"role": "user", "content": self.planning_prompt.format(instruction=instruction_list[0])},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_len = inputs["input_ids"].shape[-1]

        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

            output = self.tokenizer.decode(generation, skip_special_tokens=True)
        answer_string = '[Answer] '
        if answer_string in output:
            output = output[output.rfind(answer_string)+len(answer_string):]
        return output.split('|')
    
    def verify_subgoal(self, subgoal, current_observation_path):
        return super().verify_subgoal(subgoal, current_observation_path)
    
class QwenPlanAgent(PlanAgent):
    def __init__(self):
        super().__init__()
        model_id = "Qwen/Qwen3-14B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.planning_prompt = PLANNING_PROMPT
        self.validation_prompt = VALIDATION_PROMPT
    
    def get_plan(self, instruction_list: List[str]) -> List[str]:
        messages = [
            {"role": "user", "content": self.planning_prompt.format(instruction=instruction_list[0])},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_len = inputs["input_ids"].shape[-1]

        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

            output = self.tokenizer.decode(generation, skip_special_tokens=True)
        answer_string = '[Answer] '
        if answer_string in output:
            output = output[output.rfind(answer_string)+len(answer_string):]
        return output.split('|')
    
    def verify_subgoal(self, subgoal, current_observation_path):
        return super().verify_subgoal(subgoal, current_observation_path)
        
