import io
import json
import re
import sys
import torch
import logging
import argparse
from typing import List, Dict

import openai
import requests
import torch.nn.functional as F
from website import webshopEnv
from bs4 import BeautifulSoup
from bs4.element import Comment
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline
)


WEBSHOP_LOACTION = "http://127.0.0.1:"

# change actionly to react to run GA-rollback+ReAct
with open("../prompts/webshop/webshop_ssr_actionly_prompt.txt", 'r') as f:
    BASE_PROMPT = f.read()

with open("../prompts/webshop/webshop_ssr_analyze_examples.txt", 'r') as f:
    ANALYZE_EXAMPLE = f.read()

with open("../prompts/webshop/webshop_ssr_repetition_examples.txt", 'r') as f:
    REPETITION_EXAMPLE = f.read()

openai.api_key = ''
prob_threshold = 0.93
prob_list = []
logsum_list = []

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
            prob_list.append(average_prob)
            logsum_list.append(logsumexp_prob)
            print(average_prob)
            sys.stdout.flush()
            
            if stop:
                generate_text = [text for text in generate_text.split(stop) if text][0]
            response['text'] = generate_text.strip(' ')
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
            response['text'] = generate_text.strip(' ')
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


def get_base_query(base_query: str, start_info: str, exp: List[str]) -> str:
    query = "Task Example:\n" + base_query

    # add memory if it exists
    if len(exp) > 0:
        query += '\n\nHere is your past experiences with the current task:'
        for i, m in enumerate(exp):
            query += f'\n** Experience {i} **:\n{m.strip()}'
    query += f"\nHere is the task:{start_info}"
    return query


def generate_analysis_query(scenario: str, exp: List[str], few_shot_examples: str) -> str:
    query: str = f"""{few_shot_examples}"""
    query += f"\n# Current Task\n### Trajectory{scenario}\n##Your analysis of the current trajectory\n"
    
    return query


def format_text(text:str) -> str:
    start_pos = text.find(':')
    if start_pos > 4:
        start_pos = -1
    start_pos += 1

    last_bracket_pos = text.find(']', start_pos)
    if last_bracket_pos != -1:
        text = text[start_pos:last_bracket_pos + 1].strip()
    else:
        last_pos = text.find('\n', start_pos)
        if last_pos == -1:
            text = text[start_pos:].strip()
        else:
            text = text[start_pos:last_pos].strip()
    return text


def split_analysis(text):
    lines = text.split('\n')
    content = []
    for line in lines:
        if 'Error Location **' in line:
            loc = line[2:].strip()
            content.append(loc)
            break

    for line in lines:
        if 'Explanation **' in line:
            anal = line[2:].strip()
            content.append(anal)
            break
    
    return content 


class IntraHistory:
    def __init__(self, base_query: str, start_info, ) -> None:
        self.base_query = base_query
        self.start_info = start_info
        self.history: Dict = {
            'action_history': [],
            'obs_history': [],
        }
        self.history_len: int = 0
        self.last_action: str = ''
        self.is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation']
        if label == 'action':
            self.history['action_history'].append(value)
            if value == self.last_action:
                self.is_exhausted = True
            else:
                self.last_action = value
        elif label == 'observation':
            self.history['obs_history'].append(value)
            assert len(self.history['obs_history']) == len(self.history['action_history'])
            self.history_len += 1

    def check_is_exhausted(self) -> bool:
        return self.is_exhausted

    def reset(self) -> None:
        self.history: Dict = {
            'action_history': [],
            'obs_history': [],
        }

    def gen_query(self, exp: List[str], is_analyze=False) -> str:
        s: str = self.start_info + '\n\n' if is_analyze else get_base_query(self.base_query, self.start_info, exp) + '\n\n'
        for i in range(self.history_len):
            action = self.history['action_history'][i]
            s += f'Action {i + 1}: {action}\n'
            obs = self.history['obs_history'][i]
            s += f'Observation {i + 1}: {obs}\n\n'
        if len(self.history['action_history']) > self.history_len:
            s += f'Action {self.history_len + 1}: {self.last_action}\n'
        return s

    def rollback_history(self, roll_num) -> None:
        point = self.history_len - roll_num
        if point < 0:
            point = 0
        self.history['action_history'] = self.history['action_history'][:point]
        self.history['obs_history'] = self.history['obs_history'][:point]
        self.history_len = len(self.history['obs_history'])
        self.last_action = self.history['action_history'][-1] if self.history['action_history'] else ''
        self.is_exhausted = False

    def get_actions(self, end_point=0) -> List:
        return self.history['action_history'][:end_point] if end_point else self.history['action_history']


def gen_thought_parse(env_history, llm, exp: List, error_type=''):
    assert error_type in ['', 'repetition']
    if error_type == 'repetition':
        analyze_query = generate_analysis_query(env_history.gen_query([], True), exp, REPETITION_EXAMPLE)
    else:
        analyze_query = generate_analysis_query(env_history.gen_query([], True), exp, ANALYZE_EXAMPLE)

    max_attempts = 2
    for _ in range(max_attempts):
        response = llm._generate(analyze_query, max_new_tokens=500, gen_logits=True)
        analysis = response['text'].lstrip(' ')
        print('**************Analysis**************\n' + analysis)
        print('************************************\n')
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
                
                return earliest_e_loc, e_anal 

        print(f'Attempt {_ + 1}: Failed to generate correct format.')
        sys.stdout.flush()
    raise ValueError(f'Failed to generate content in the correct format after {max_attempts} attempts')


def rollback(idx: str, env, env_history, llm, e_loc: int, exp: List):
    rollback_num = env_history.history_len - e_loc + 1
    if rollback_num > 3:
        rollback_num = 3
    if rollback_num < 1:
        rollback_num = 1
    env_history.rollback_history(rollback_num)

    act_list = env_history.get_actions()
    # restore to the state before the error
    _ = env.step(idx, 'reset')

    for action in act_list:
        _ = env.step(idx, action)

    new_query = env_history.gen_query(exp) + f'Action {env_history.history_len + 1}:'

    response = llm._generate(new_query[-(6400 - len(env_history.base_query)):], stop='\n')
    new_action = response['text'].lstrip(' ')
    new_action = format_text(new_action)

    return env, env_history, rollback_num, new_action, new_query


def webshop_run(idx, env, base_prompt, llm, assist_llm, is_assist, to_print=True, max_rollback_num=6):
    action = 'reset'
    exp, rollback_record = [], []
    rollback_times = 0

    res = env.step(idx, action)
    observation = res[0]

    if is_assist:
        assistant = assist_llm
    else:
        assistant = llm
    
    # Task environment history
    env_history = IntraHistory(base_prompt, observation)
    if to_print:
        print(f'Action 0: {action}\nObservation 0: {observation}\n')
        sys.stdout.flush()
    
    i = 0
    while True:
        query = env_history.gen_query(exp) + f'Action {env_history.history_len + 1}:'
        response = llm._generate(query[-(6400 - len(base_prompt)):], stop='\n')
        action = response['text'].lstrip(' ')
        action = format_text(action)
        
        try:
            res = env.step(idx, action)
            observation = res[0]
        except AssertionError:
            observation = 'Invalid action!'

        if action.startswith('think'):
            observation = 'OK.'
        
        env_history.add('action', action)
        env_history.add('observation', observation)
        tag = False

        # check if current action needs rollback
        if rollback_times < max_rollback_num:
            try:
                if env_history.check_is_exhausted():
                    e_loc, e_anal = gen_thought_parse(env_history, assistant, exp, 'repetition')
                else:
                    e_loc, e_anal = gen_thought_parse(env_history, assistant, exp)

                if e_loc:
                    # error detected, rollback and regenerate
                    exp.append(e_anal)
                    exp = exp[-3:]
                    env, env_history, rollback_num, action, n_query = rollback(idx, env, env_history, llm, e_loc, exp)
                    rollback_times += 1
                    
                    try:
                        res = env.step(idx, action)
                        observation = res[0]
                    except AssertionError:
                        observation = 'Invalid action!'

                    if action.startswith('think'):
                        observation = 'OK.'
                    
                    env_history.add('action', action)
                    env_history.add('observation', observation)
                    i = env_history.history_len
                    
                    roll_log = {
                        'id': rollback_times,
                        'rollback_steps': rollback_num,
                        'new_query': n_query,
                        'new_action': action
                    }
                    rollback_record.append(roll_log)
                    tag = True

            except ValueError as e:
                print(e)
                sys.stdout.flush()

        if to_print:
            if tag:
                print('---rollback happened---')
                sys.stdout.flush()
            else:
                i += 1
            print(f'Action {i}: {action}\nObservation {i}: {observation}\n')
            sys.stdout.flush()

        if res[2]:
            return res[1], rollback_record

        if env_history.history_len >= 15:
            return 0, rollback_record


def run_episodes(prompt, webshop_url, llm, assist_llm=None, is_assist=False, n=50, max_rollback_num=6):
    rs, roll_logs, rollback_record = [], [], []
    cnt = 0
    env = webshopEnv(webshop_url)
    for i in range(n):
        print(i)
        sys.stdout.flush()
        try:
            r, rollback_record = webshop_run(f'fixed_{i}', env, prompt, llm, assist_llm, is_assist, True, max_rollback_num)
        except AssertionError:
            r = 0
            cnt += 1
        rs.append(r)
        record_dict = {
            'task_id': f'fixed_{i}',
            'reward': r,
            'is_success': r == 1,
            'rollback_record': rollback_record
        }
        roll_logs.append(record_dict)
        if (i + 1) % 1 == 0:
            r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
            print(f"id:{i + 1}, score:{r}, success rate:{sr}, false rate{fr}")
            print('---------------------------------------------------------------------------')
            sys.stdout.flush()
    r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
    print(f"score:{r}, success rate:{sr}, false rate{fr}")
    sys.stdout.flush()
    return rs, roll_logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=5000, help="your port of deployed webshop")
    parser.add_argument("--model_source", type=str, default="open", choices=["open", "close"])
    parser.add_argument("--max_roll_num", type=int, default=6, help="maximum number of rollbacks")
    parser.add_argument("--sample_num", type=int, default=100, help="number of test instances")
    parser.add_argument("--llm_name_or_path", type=str, help="name of closed LLM or path of open-sourced LLM used for generator")
    parser.add_argument("--force_same_LM", type=int, default=1, help="force generator and assistant to be the same model.")
    parser.add_argument("--assist_name_or_path", type=str, default='None', help="Name or path of the LLM used as Assitant")
    parser.add_argument("--out_record_path", type=str, help="the file path for saving rollback records")
    args = parser.parse_args()
    
    WEBSHOP_URL = WEBSHOP_LOACTION + args.port
    
    if args.model_source == 'open':
        llm = Local_llm(args.llm_name_or_path)
    else:
        llm = Api_llm(args.llm_name_or_path)
    
    if not args.force_same_LM:
        if args.model_source == 'open':
            assist_llm = Local_llm(args.assist_name_or_path)
        else:
            assist_llm = Api_llm(args.assist_name_or_path)
        res, roll_logs = run_episodes(BASE_PROMPT, WEBSHOP_URL, llm, assist_llm, True, args.sample_num, args.max_roll_num)
    else:
        res, roll_logs = run_episodes(BASE_PROMPT, WEBSHOP_URL, llm, n=args.sample_num, max_rollback_num=args.max_roll_num)
    
    json.dump(roll_logs, open(args.out_record_path, 'w'), indent=4)
