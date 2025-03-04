import os
import sys
import json
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from typing import Any, List, Dict, Tuple
from game24 import Game24Task

    
file_path = '24.csv'
FOLDER = '../prompts/game24/'
ACTIONLY_FILE = 'game24_base_actionly.txt'
REACT_FILE = 'game24_base_react.txt'


class Local_llm:
    def __init__(self, local_path):
        self.name = local_path.split('/')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        self.model.half()
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        
        self.generation_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            device_map="auto"
        )
    
    def _generate(self, prompt: str, stop=None):
        sequences = self.generation_pipe(
            prompt,
            temperature=0.1,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
            top_p=1,
            do_sample=True
        )
        # print('########################\n' + prompt + '\n########################')
        # sys.stdout.flush()
        action = sequences[0]["generated_text"]
        if stop:
            action = action.split(stop)[0]

        if len(action) > 1:
            action = action[0].lower() + action[1:]

        return action.strip()


def game24_run(task_input, env, prompt, llm, to_print=True):
    ob = '\n# Here is the task:\nInput: ' + task_input
    init_prompt = prompt + ob + '\nAct 1> '
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 12):
        action = llm._generate(init_prompt + prompt, stop='\n')
        response = env.step(action)
        
        reward, observation = response['r'], response['ob']
        if to_print:
            print(f'Act {i}> {action}\nObs {i}> {observation}')
            sys.stdout.flush()
        prompt += f'{action}\nObs {i}> {observation}\nAct {i+1}> '
        if reward:
            return reward
        elif 'Exceeded' in observation:
            return 0
    return 0


def main(args):
    env = Game24Task(file_path, args.seed, args.num_samples)
    if args.mode == 'react':
        with open(os.path.join(FOLDER, REACT_FILE), 'r') as f:
            BASE_PROMPT = f.read()
    elif args.mode == 'act':
        with open(os.path.join(FOLDER, ACTIONLY_FILE), 'r') as f:
            BASE_PROMPT = f.read()
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Mode must be 'react' or 'act'")
    
    llm = Local_llm(args.local_llm_path)
    results = []
    
    for i in range(args.num_samples):
        print('---------------------------------------------------------------------------')
        sys.stdout.flush()
        curr_case = env.get_case()
        r = game24_run(curr_case, env, BASE_PROMPT, llm)
        results.append(r)
        if (i + 1) % 1 == 0:
            sr = sum(results) / len(results)
            print(f"id:{i + 1}, success rate:{sr}")
            print('---------------------------------------------------------------------------')
            sys.stdout.flush()

    print(f"Success rate:{sum(results) / len(results)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, help="The number of samples")
    parser.add_argument("--seed", type=int, help="Sample seed")
    parser.add_argument("--local_llm_path", type=str, help="local path of utilized LLM")
    parser.add_argument("--mode", type=str, default='react', help="act / react")
    args = parser.parse_args()
    
    main(args)
