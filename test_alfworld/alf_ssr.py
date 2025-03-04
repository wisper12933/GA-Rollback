import os
import json
import sys
import re
import argparse
from copy import deepcopy

import openai
import yaml
import alfworld
import torch
import torch.nn.functional as F
import alfworld.agents.environment
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open("../prompts/alfworld/alfworld_ssr_base_prompts.json", 'r') as f:
    d = json.load(f)

with open("../prompts/alfworld/alfworld_analyze_examples.txt", 'r') as f:
    ANALYZE_EXAMPLE = f.read()

with open("../prompts/alfworld/alfworld_repetition_examples.txt", 'r') as f:
    REPETITION_EXAMPLE = f.read()

openai.api_key = ''
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
        # self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        # print(f'Using device: {self.device}')
        
        # self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     local_path,
        #     torch_dtype=torch.float16,
        #     trust_remote_code=True
        # ).to(self.device)
        
        # self.model.half()
        # self.model.eval()
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     self.model = torch.compile(self.model)
        
        # self.generation_pipe = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     trust_remote_code=True,
        #     device=self.device.index
        # )
    
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
        client = OpenAI(api_key="", base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            stop=stop,
            temperature=0.0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        # response = openai.Completion.create(
        #   model="text-davinci-002",
        #   prompt=prompt,
        #   temperature=0,
        #   max_tokens=100,
        #   top_p=1,
        #   frequency_penalty=0.0,
        #   presence_penalty=0.0,
        #   stop=stop
        # )
        action = response.choices[0].message.content
        action = extract_substring(generate_text)
        if len(action) > 1:
            action = action[0].lower() + action[1:]
        
        return action


def get_base_query(base_query: str, start_info: str, exp: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(exp) > 0:
        query += '\n\nYour past experiences with the current task below:'
        for i, m in enumerate(exp):
            query += f'\n** Experience {i} **:\n{m.strip()}'
    query += f"\n{start_info}"
    return query


def generate_analysis_query(scenario: str, exp: List[str], few_shot_examples: str) -> str:
    query: str = f"""{few_shot_examples}"""

    # if len(exp) > 0:
    #     query += '\n\nAnalysis from past attempts:\n'
    #     for i, m in enumerate(exp):
    #         query += f'Trial #{i}: {m}\n'

    query += f"\n# Current Task\n### Trajectory{scenario}\n##Your analysis of the current trajectory\n"
    return query


def format_text(text:str) -> str:
    start_pos = text.find('>')
    return text[start_pos + 1:].strip()


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
    def __init__(self, base_query: str, start_info: str) -> None:
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
        s: str = self.start_info + '\n' if is_analyze else get_base_query(self.base_query, self.start_info, exp) + '\n'
        for i in range(self.history_len):
            action = self.history['action_history'][i]
            s += f'Act {i + 1}> {action}\n'
            obs = self.history['obs_history'][i]
            s += f'Obs {i + 1}> {obs}\n'
        if len(self.history['action_history']) > self.history_len:
            s += f'Act {self.history_len + 1}> {self.last_action}\n'
        return s

    def rollback_history(self, roll_num) -> None:
        self.history['action_history'] = self.history['action_history'][:self.history_len - roll_num]
        self.history['obs_history'] = self.history['obs_history'][:self.history_len - roll_num]
        self.history_len = len(self.history['obs_history'])
        self.last_action = self.history['action_history'][-1] if self.history['action_history'] else ''
        self.is_exhausted = False

    def get_actions(self, end_point=0) -> List:
        return self.history['action_history'][:end_point] if end_point else self.history['action_history']


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


def gen_thought_parse(env_history, llm, exp: List, error_type=''):
    assert error_type in ['', 'repetition']
    if error_type == 'repetition':
        analyze_query = generate_analysis_query(env_history.gen_query([], True), exp, REPETITION_EXAMPLE)
    else:
        analyze_query = generate_analysis_query(env_history.gen_query([], True), exp, ANALYZE_EXAMPLE)

    max_attempts = 2
    for _ in range(max_attempts):
        response = llm._generate(analyze_query, max_new_tokens=800, gen_logits=True)
        analysis = response['text'].lstrip(' ')
        # print('**************Analysis Query**************\n' + analyze_query)
        # print('******************************************')
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
                
                # experience = env_history.gen_query([], True)
                # experience = experience + 'Analysis:' + e_anal
                
                return earliest_e_loc, e_anal  # loc, analysis

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
    env.batch_env.reset()
    
    for action in act_list:
        _ = env.step([action])

    new_query = env_history.gen_query(exp) + f'Act {env_history.history_len + 1}>'
    # print('\n**************New Query**************\n' + new_query)
    # print('*************************************\n\n')
    # sys.stdout.flush()
    response = llm._generate(new_query, stop='\n')
    new_action = response['text'].lstrip(' ')
    new_action = format_text(new_action)

    return env, env_history, rollback_num, new_action, new_query


def alfworld_run(env, prompt, llm, assist_llm, is_assist, to_print=True, ob='', max_rollback_num=6, wait_k=6):
    if to_print:
        print(ob)
        sys.stdout.flush()
    
    if is_assist:
        assistant = assist_llm
    else:
        assistant = llm
    
    exp, rollback_record = [], []
    rollback_times, i = 0, 0
    
    
    # Task environment history
    env_history = IntraHistory(prompt, ob)
    
    while True:
        query = env_history.gen_query(exp) + f'Act {env_history.history_len + 1}>'
        response = llm._generate(query, stop='\n')
        action = response['text'].strip()
        action = format_text(action)
        
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        
        env_history.add('action', action)
        env_history.add('observation', observation)
        roll_tag = False
        
        # check if current action needs rollback
        if rollback_times < max_rollback_num and env_history.history_len > wait_k:
            try:
                if env_history.check_is_exhausted():
                    e_loc, e_anal = gen_thought_parse(env_history, assistant, exp, 'repetition')
                else:
                    e_loc, e_anal = gen_thought_parse(env_history, assistant, exp)
                
                if e_loc:
                    # error detected, rollback and regenerate
                    exp.append(e_anal)
                    exp = exp[-3:]
                    env, env_history, rollback_num, action, n_query = rollback(env, env_history, llm, e_loc, exp)
                    rollback_times += 1
                    
                    observation, reward, done, info = env.step([action])
                    observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
                    if action.startswith('think:'):
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
        
        if done:
            return reward, rollback_record
        if env_history.history_len > 50:
            return 0, rollback_record


prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}


def run_tasks(llm, assist_llm=None, is_assist=False, max_rollback_num=6, wait_k=6):
    cnts = [0] * 6
    rs = [0] * 6
    roll_logs = []
    
    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    
    split = "eval_out_of_distribution"
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    for idx in range(134):
        ob, info = env.reset()
        print(f'Show: ob={ob}')
        print(f'Show: info={info}')
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(name)
        print(f'*****************************************************\nBegin Task {idx + 1}\n*****************************************************')
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                # mode react/act
                prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'act_{v}_1'] + d[f'act_{v}_0']
                print(k, v)
                
                r, rollback_record = alfworld_run(env, prompt, llm, assist_llm, is_assist, ob=ob, max_rollback_num=max_rollback_num, wait_k=wait_k)

                rs[i] += r
                cnts[i] += 1
                break
        
        record_dict = {
            'task_id': f'{idx + 1}',
            'is_success': r,
            'rollback_record': rollback_record
        }
        roll_logs.append(record_dict)
        print(idx + 1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
        print('------------\n')
    return roll_logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_source", type=str, default="open", choices=["open", "close"])
    parser.add_argument("--max_roll_num", type=int, default=6, help="maximum number of rollbacks")
    parser.add_argument("--llm_name_or_path", type=str, help="name of closed LLM or path of open-sourced LLM")
    parser.add_argument("--force_same_LM", type=int, default=1, help="force generator and assistant to be the same model.")
    parser.add_argument("--assist_name_or_path", type=str, default='None', help="Name or path of the LLM used as Assitant")
    parser.add_argument("--out_record_path", type=str, help="the file path for saving rollback records")
    parser.add_argument("--wait_k", type=int, help="the file path for saving rollback records")
    args = parser.parse_args()
    
    if args.model_source == 'open':
        llm = Local_llm(args.llm_name_or_path)
    else:
        llm = Api_llm(args.llm_name_or_path)
    
    if not args.force_same_LM:
        if args.model_source == 'open':
            assist_llm = Local_llm(args.assist_name_or_path)
        else:
            assist_llm = Api_llm(args.assist_name_or_path)
        roll_logs = run_tasks(llm, assist_llm, True, args.max_roll_num, wait_k=args.wait_k)
    else:
        roll_logs = run_tasks(llm, max_rollback_num=args.max_roll_num, wait_k=args.wait_k)
    json.dump(roll_logs, open(args.out_record_path, 'w'), indent=4)
