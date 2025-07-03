import os
import json
import sys
import re
import argparse

import torch
import torch.nn.functional as F
from game24 import Game24Task, IntraHistory
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "true"

file_path = '24.csv'
FOLDER = '../prompts/game24/'
ACTIONLY_FILE = 'game24_base_actionly.txt'
REACT_FILE = 'game24_base_react.txt'

with open("../prompts/game24/game24_analyze_examples.txt", 'r') as f:
    ANALYZE_EXAMPLE = f.read()

prob_threshold = 0.93


class Local_llm:
    def __init__(self, local_path):
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
    
    def _generate(self, prompt: str, stop=None, max_new_tokens=100, gen_logits=False):  
        response = {
            'text': None,
            'average_prob': None,
            'logsumexp_prob': None
        }
        if gen_logits:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, 'device_map'):
                first_device = next(iter(self.model.device_map.values()))
                inputs = inputs.to(f'cuda:{first_device}' if isinstance(first_device, int) else first_device)
            else:
                inputs = inputs.to(self.model.device)
                
            outputs = self.model.generate(
                **inputs,
                temperature=0.1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                top_p=1,
                do_sample=True
            )
            generate_sequence = outputs.sequences[0]
            generate_tokens = generate_sequence[inputs.input_ids.shape[1]:]
            generate_text = self.tokenizer.decode(generate_tokens, skip_special_tokens=True)
            
            logits = torch.stack(outputs.scores, dim=0)
            probs = F.softmax(logits, dim=-1)

            selected_probs = probs[range(len(generate_tokens)), 0, generate_tokens]
            average_prob = selected_probs.mean().item()
            logsumexp_prob = torch.logsumexp(selected_probs, dim=0).item()
            response['average_prob'] = average_prob
            response['logsumexp_prob'] = logsumexp_prob
            
            if stop:
                generate_text = [text for text in generate_text.split(stop) if text][0]
            response['text'] = generate_text.strip()
            return response
        else:
            sequences = self.generation_pipe(
                prompt,
                temperature=0.1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                top_p=1,
                do_sample=True
            )
            generate_text = sequences[0]["generated_text"]
            if stop:
                generate_text = [text for text in generate_text.split(stop) if text][0]
            response['text'] = generate_text.strip()
            return response


class Api_llm:
    def __init__(self, name):
        self.model_name = name
        
    def _generate(self, prompt, stop=None, max_new_tokens=100):
        stops = []
        if stop:
            stops.append(stop)
        response = openai.ChatCompletion.create(
            model=self.name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=max_new_tokens,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stops
        )
        return response["choices"][0]["message"]["content"]


def generate_analysis_query(scenario: str, exp: List[str], few_shot_examples: str) -> str:
    query: str = f"""{few_shot_examples}"""

    query += f"\n# Current Task\n## Trajectory{scenario}\n##Your analysis of the current trajectory\n"
    return query


def format_text(text:str) -> str:
    start_pos = text.find('>')
    return text[start_pos + 1:].strip()


def split_analysis(text):
    lines = text.split('\n')
    content = []
    for line in lines:
        if 'Error Location' in line:
            loc = line[2:].strip()
            content.append(loc)
            break

    for line in lines:
        if 'Explanation' in line:
            anal = line[2:].strip()
            content.append(anal)
            break
    
    return content 


def gen_thought_parse(env_history, llm, exp: List):
    # Todo Analysis Example
    analyze_query = generate_analysis_query(env_history.gen_query([], True), exp, ANALYZE_EXAMPLE)

    max_attempts = 2
    for _ in range(max_attempts):
        response = llm._generate(analyze_query, max_new_tokens=500, gen_logits=True)
        analysis = response['text'].lstrip(' ')
        print('**************Analysis**************\n' + analysis)
        print('************************************')
        sys.stdout.flush()
        
        # Analytical quality assessment ave/logsumexp
        ave_probs = response['average_prob']
        if ave_probs and ave_probs < prob_threshold:
            return 0, ''
        
        contents = split_analysis(analysis)
        if len(contents) == 2:
            if 'None' in contents[0] or 'no error' in contents[1]:
                # correct
                return 0, ''
            else:
                # parse
                # earliest error location
                st_pos = contents[0].find(':')
                e_loc = contents[0][st_pos + 1:].strip()
                nums = re.findall(r'\d+', e_loc)
                nums = [int(num) for num in nums]
                earliest_e_loc = min(nums)

                st_pos = contents[1].find(':')
                e_anal = contents[1][st_pos + 1:].strip()
                sents = e_anal.split('.')
                if sents[-1].strip():
                    sents = sents[:-1]
                e_anal = '.'.join(sents) + '.'
                
                experience = env_history.gen_query([], True)
                experience = experience + 'Analysis:' + e_anal
                
                return earliest_e_loc, experience

        print(f'Attempt {_ + 1}: Failed to generate correct format.')
        sys.stdout.flush()
    raise ValueError(f'Failed to generate content in the correct format after {max_attempts} attempts')


def rollback(env, env_history, llm, e_loc: int, exp: List):
    rollback_num = env_history.history_len - e_loc + 1
    if rollback_num > 4:
        rollback_num = 4
    if rollback_num < 1:
        rollback_num = 1
    env_history.rollback_history(rollback_num)

    act_list = env_history.get_actions()
    env.get_case(reset=True)
    
    for action in act_list:
        _ = env.step(action)

    new_query = env_history.gen_query(exp) + f'Act {env_history.history_len + 1}>'

    response = llm._generate(new_query, stop='\n')
    new_action = response['text'].lstrip(' ')
    new_action = format_text(new_action)

    return env, env_history, rollback_num, new_action, new_query


def game24_run(task_input, env, prompt, llm, assist_llm, max_rollback_num, to_print=True):
    ob = '\n# Here is the task:\nInput: ' + task_input
    env_history = IntraHistory(prompt, ob)
    if to_print:
        print(ob)
        sys.stdout.flush()
        
    exp = []
    rollback_times, i = 0, 0
        
    while True:
        query = env_history.gen_query(exp) + f'Act {env_history.history_len + 1}>'
        response = llm._generate(query, stop='\n')
        action = response['text'].strip()
        action = format_text(action)
        
        response = env.step(action)
        reward, observation = response['r'], response['ob']
        
        env_history.add('action', action)
        env_history.add('observation', observation)
        roll_tag = False
        
        # check if current action needs rollback
        if rollback_times < max_rollback_num:
            try:
                e_loc, e_anal = gen_thought_parse(env_history, assist_llm, exp)
                
                if e_loc:
                    # error detected, rollback and regenerate
                    exp.append(e_anal)
                    exp = exp[-3:]
                    env, env_history, rollback_num, action, n_query = rollback(env, env_history, llm, e_loc, exp)
                    rollback_times += 1
                    
                    response = env.step(action)
                    reward, observation = response['r'], response['ob']
                    
                    env_history.add('action', action)
                    env_history.add('observation', observation)
                    i = env_history.history_len
                    
                    roll_tag = True
                    
            except ValueError as e:
                print(e)
                sys.stdout.flush()
        
        if to_print:
            if roll_tag:
                print('---rollback happened---')
                sys.stdout.flush()
            else:
                i += 1
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        
        if reward:
            return reward
        elif 'Exceeded' in observation:
            return 0
        if env_history.history_len > 11:
            return 0


def run_tasks(args):
    if args.model_source == 'open':
        llm = Local_llm(args.llm_name_or_path)
    else:
        llm = Api_llm(args.llm_name_or_path)
    
    if not args.force_same_LM:
        if args.model_source == 'open':
            assist_llm = Local_llm(args.assist_name_or_path)
        else:
            assist_llm = Api_llm(args.assist_name_or_path)
    else:
        assist_llm = llm
        
    if args.mode == 'react':
        with open(os.path.join(FOLDER, REACT_FILE), 'r') as f:
            BASE_PROMPT = f.read()
    elif args.mode == 'act':
        with open(os.path.join(FOLDER, ACTIONLY_FILE), 'r') as f:
            BASE_PROMPT = f.read()
    
    results = []
    env = Game24Task(file_path, args.seed, args.num_samples)
    for i in range(args.num_samples):
        print('---------------------------------------------------------------------------')
        sys.stdout.flush()
        curr_case = env.get_case()
        r = game24_run(curr_case, env, BASE_PROMPT, llm, assist_llm, args.max_roll_num)
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
    parser.add_argument("--model_source", type=str, default="open", choices=["open", "close"])
    parser.add_argument("--max_roll_num", type=int, default=6, help="maximum number of rollbacks")
    parser.add_argument("--llm_name_or_path", type=str, help="name of closed LLM or path of open-sourced LLM")
    parser.add_argument("--force_same_LM", type=int, default=1, help="force generator and assistant to be the same model.")
    parser.add_argument("--assist_name_or_path", type=str, default='None', help="Name or path of the LLM used as Assitant")
    parser.add_argument("--mode", type=str, default='act', help="act / ract", choices=["act", "react"])
    args = parser.parse_args()
    
    run_tasks(args)
