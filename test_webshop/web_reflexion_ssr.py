import os
import re
import sys
import json
import yaml
import torch
import requests
import argparse

from bs4 import BeautifulSoup
from bs4.element import Comment
from website import webshopEnv
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig, 
    pipeline)
import torch.nn.functional as F
from typing import Any, List, Dict, Tuple


ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

with open("../prompts/webshop/webshop_ssr_actionly_prompt.txt", 'r') as f:
    BASE_PROMPT = f.read()

with open("../prompts/webshop/webshop_ssr_analyze_examples.txt", 'r') as f:
    ANALYZE_EXAMPLE = f.read()

with open("../prompts/webshop/webshop_ssr_repetition_examples.txt", 'r') as f:
    REPETITION_EXAMPLE = f.read()

with open("../prompts/webshop/webshop_reflexion_examples.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

prob_threshold = 0.93


class Local_llm:
    def __init__(self, local_path, device_id):
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        
        self.model.half()
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        
        self.generation_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            device=self.device.index
        )
    
    def _generate(self, prompt: str, stop=None, max_new_tokens=100, gen_logits=False):
        response = {
            'text': None,
            'average_prob': None,
            'logsumexp_prob': None
        }
        if gen_logits:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
            # print(average_prob)
            # sys.stdout.flush()
            
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
                # top_p=1,
                # do_sample=True
            )
            generate_text = sequences[0]["generated_text"]
            if stop:
                generate_text = [text for text in generate_text.split(stop) if text][0]
            response['text'] = generate_text.strip(' ')
            return response


class IntraHistory:
    def __init__(self, base_query: str, start_info, memory: List[str]) -> None:
        self.base_query = base_query
        self.start_info = start_info
        self.memory = memory
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
        s: str = self.start_info + '\n\n' if is_analyze else get_base_query(self.base_query, self.start_info,
                                                                            exp, self.memory) + '\n\n'
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
    
    def __str__(self) -> str:
        cur_query = self.base_query
        if len(self.memory) > 0:
            cur_query += '\nYour memory for the task below:'
            for i, m in enumerate(self.memory):
                mem = m['text'].strip()
                cur_query += f'\nTrial {i}:\n{mem}'
        cur_query += f"\nHere is the task:\n{self.start_info}"
        
        s: str = cur_query + '\n'
        for i in range(self.history_len):
            action = self.history['action_history'][i]
            s += f'Action {i + 1}: {action}\n'
            obs = self.history['obs_history'][i]
            s += f'Observation {i + 1}: {obs}\n\n'
        return s


def extract_substring(s: str) -> str:
    start_pos = s.find(':')
    if start_pos > 4:
        start_pos = -1
    end_pos = s.find(']', start_pos)
    if end_pos != -1:
        return s[start_pos + 1: end_pos + 1].strip()
    else:
        return s[start_pos + 1:].strip()


def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    return s.split("Instruction:")[-1].strip()


def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. There are two examples below.

{FEW_SHOT_EXAMPLES}

Instruction: {scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            mem = m['text']
            query += f'Trial #{i}: {mem}\n'

    query += "\n\nNew plan:"
    return query


def generate_analysis_query(scenario: str, exp: List[str], few_shot_examples: str) -> str:
    query: str = f"""{few_shot_examples}"""

    # if len(exp) > 0:
    #     query += '\n\nAnalysis from past attempts:\n'
    #     for i, m in enumerate(exp):
    #         query += f'Trial #{i}: {m}\n'

    query += f"\n# Current Task\n### Trajectory{scenario}\n##Your analysis of the current trajectory\n"
    return query


def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]], model) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    # changed
    with open(trial_log_path, 'r', encoding='utf-8') as f:
        full_log: str = f.read()
        
    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            reflection_query: str = _generate_reflection_query(env_logs[i], memory)
            reflection: str = model._generate(reflection_query, '\n') # type: ignore
            env_configs[i]['memory'] += [reflection]
                
    return env_configs


def get_base_query(base_query: str, start_info: str, exp: List[str], memory: List[str]) -> str:
    query = "Task Example:\n" + base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour plan memory for the task below:'
        for i, m in enumerate(memory):
            mem = m['text'].strip()
            query += f'\n** Trial {i} **:\n{mem}'
            
    if len(exp) > 0:
        query += '\n\nHere is your past experiences with the current task:'
        for i, m in enumerate(exp):
            query += f'\n** Experience {i} **:\n{m.strip()}'
            
    query += f"\nHere is the task:\n{start_info}"
    return query


def format_text(text:str) -> str:
    start_pos = text.find(':')
    if start_pos == -1:
        start_pos = 0
    else:
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
        if line.startswith('** Error Location'):
            loc = line[2:].strip()
            content.append(loc)
            break

    for line in lines:
        if line.startswith('** Explanation'):
            anal = line[2:].strip()
            content.append(anal)
            break
    
    return content 


def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )
    

def gen_thought_parse(env_history, llm, exp: List, error_type=''):
    assert error_type in ['', 'repetition']
    if error_type == 'repetition':
        analyze_query = generate_analysis_query(env_history.gen_query([], True), exp, REPETITION_EXAMPLE)
    else:
        analyze_query = generate_analysis_query(env_history.gen_query([], True), exp, ANALYZE_EXAMPLE)

    max_attempts = 2
    for _ in range(max_attempts):
        response = llm._generate(analyze_query, max_new_tokens=250, gen_logits=True)
        analysis = response['text'].lstrip(' ')
        # print('***********Analysis Query***********\n' + analyze_query)
        # print('************************************\n')
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
                # Changed, delete experience to back to last version
                # experience = env_history.gen_query([], True)
                # experience = experience + 'Analysis:' + e_anal
                
                return earliest_e_loc, e_anal  # loc, analysis

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
    # print('\n**************New Query**************\n' + new_query)
    # print('*************************************\n\n')
    # sys.stdout.flush()
    response = llm._generate(new_query[-(6400 - len(env_history.base_query)):], stop='\n')
    new_action = response['text'].lstrip(' ')
    new_action = format_text(new_action)

    return env, env_history, rollback_num, new_action, new_query


def webshop_run(idx, env, base_prompt, memory: List[str], model, assist_model, max_rollback_num=6, to_print=True):
    action = 'reset'
    exp, rollback_record = [], []
    rollback_times = 0

    res = env.step(idx, action)
    observation = res[0]
    if len(memory) > 3:
        env_history = IntraHistory(base_prompt, observation, memory[-3:])
    else:
        env_history = IntraHistory(base_prompt, observation, memory)
    env_history.reset()
    if to_print:
        print(f'Action 0: {action}\nObservation 0: {observation}\n')
        sys.stdout.flush()
    
    i = 0
    while True:
        query = env_history.gen_query(exp) + f'Action {env_history.history_len + 1}:'
        response = model._generate(query[-(6400 - len(base_prompt)):], stop='\n')
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
                    e_loc, e_anal = gen_thought_parse(env_history, assist_model, exp, 'repetition')
                else:
                    e_loc, e_anal = gen_thought_parse(env_history, assist_model, exp)

                if e_loc:
                    # error detected, rollback and regenerate
                    exp.append(e_anal)
                    exp = exp[-3:]
                    env, env_history, rollback_num, action, n_query = rollback(idx, env, env_history, model, e_loc, exp)
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
            return env_history, res[1], rollback_record

        if env_history.history_len >= 15:
            return env_history, 0, rollback_record


def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model,
        assist_model,
        max_rollback_num,
        web_loc,
    ) -> List[Dict[str, Any]]:
    env = webshopEnv(web_loc)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)
    rs, roll_logs, rollback_record = [], [], []

    for z, env_config in enumerate(env_configs):
        if env_config["is_success"]:
            num_successes += 1
            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        try:
            final_env_history, reward, rollback_record = webshop_run(f'fixed_{z}', env, BASE_PROMPT, env_config["memory"] if use_memory else [], model, assist_model, max_rollback_num, to_print=True)
            is_success = (reward == 1.0)
            rs.append(reward)
            env_configs[z]["reward"] = reward if reward > env_configs[z]["reward"] else env_configs[z]["reward"]
            
            record_dict = {
                'task_id': f'fixed_{z}',
                'reward': reward,
                'is_success': is_success,
                'rollback_record': rollback_record
            }
            roll_logs.append(record_dict)
            
            if is_success:
                status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                env_configs[z]["is_success"] = True
                num_successes += 1
                num_additional_successes += 1
            else:
                status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

            # log env results to trial log
            # changed
            with open(trial_log_path, 'a', encoding='utf-8') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

        except AssertionError:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

            # log env results to trial log
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}:\nAssertion Error\n\nSTATUS: FAIL\n\n#####\n')

        # log to world log
        with open(world_log_path, 'a') as f:
            f.write(status_str + '\n')

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs, roll_logs


def main(args) -> None:
    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")
        logging_dir = args.resume_dir

        # load environment configs
        env_config_path: str = os.path.join(args.resume_dir, f'env_results_trial_{args.start_trial_num - 1}.json')
        if not os.path.exists(env_config_path):
            raise ValueError(f"Environment config file `{env_config_path}` does not exist")
        with open(env_config_path, 'r') as rf:
            env_configs: List[Dict[str, Any]] = json.load(rf)
    else:
        # Create the run directory
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        logging_dir = args.run_name

        # initialize environment configs
        env_configs: List[Dict[str, Any]] = []
        for i in range(args.num_envs):
            env_configs += [{
                'name': f'env_{i}',
                'memory': [],
                'reward': 0,
                'is_success': False
            }]
    
    world_log_path: str = os.path.join(logging_dir, 'world.log')

    # print start status to user
    if args.is_resume:
        print(f"""
    -----
    Resuming run with the following parameters:
    Run name: {logging_dir}
    Number of trials: {args.num_trials}
    Number of environments: {args.num_envs}
    Use memory: {args.use_memory}
    Resume trial number: {args.start_trial_num}

    Sending all logs to `{args.run_name}`
    -----
    """)
    else:
        print(f"""
    -----
    Starting run with the following parameters:
    Run name: {logging_dir}
    Number of trials: {args.num_trials}
    Number of environments: {args.num_envs}
    Use memory: {args.use_memory}

    Sending all logs to `{args.run_name}`
    -----
    """)

    # run trials
    trial_idx = args.start_trial_num
    
    if args.model_source == 'open':
        llm = Local_llm(args.llm_name_or_path, 0)
    else:
        llm = Api_llm(args.llm_name_or_path)
    
    if not args.force_same_LM:
        if args.model_source == 'open':
            assist_llm = Local_llm(args.assist_name_or_path, 1)
        else:
            assist_llm = Api_llm(args.assist_name_or_path)
    else:
        assist_llm = llm
    
    logs = {}
    
    while trial_idx < args.num_trials:
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

        # set paths to log files
        trial_log_path: str = os.path.join(args.run_name, f'trial_{trial_idx}.log')
        trial_env_configs_log_path: str = os.path.join(args.run_name, f'env_results_trial_{trial_idx}.json')
        if os.path.exists(trial_log_path):
            open(trial_log_path, 'w').close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, 'w').close()

        # run trial
        _, trial_logs = run_trial(trial_log_path, world_log_path, trial_idx, env_configs, args.use_memory, llm, assist_llm, args.max_roll_num, args.website)
        logs[f'trial_{trial_idx}_record'] = trial_logs

        # update memory if needed
        if args.use_memory:
            env_configs: List[Dict[str, Any]] = update_memory(trial_log_path, env_configs, llm)

        # log env configs for trial
        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

        # log world for trial
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')
        
        rewards = []
        for env_config in env_configs:
            rewards.append(env_config['reward'])
        print(f'***** Trial #{trial_idx} Average Reward :{sum(rewards)/len(rewards)}')
        trial_idx += 1
    json.dump(logs, open(args.out_record_path, 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, help="The number of trials to run")
    parser.add_argument("--num_envs", type=int, help="The number of environments per trial")
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--use_memory", action='store_true', help="Allow the Agent to use memory")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
    parser.add_argument("--model_source", type=str, default="open", choices=["open", "close"])
    parser.add_argument("--llm_name_or_path", type=str, help="name of closed LLM or path of open-sourced LLM used for generator")
    parser.add_argument("--max_roll_num", type=int, default=6, help="maximum number of rollbacks")
    parser.add_argument("--force_same_LM", type=int, default=1, help="force generator and assistant to be the same model.")
    parser.add_argument("--assist_name_or_path", type=str, default='None', help="Name or path of the LLM used as Assitant")
    parser.add_argument("--out_record_path", type=str, help="the file path for saving rollback records")
    parser.add_argument("--website", type=str, help="webshop_url")
    args = parser.parse_args()

    assert args.num_trials > 0, "Number of trials should be positive"
    assert args.num_envs > 0, "Number of environments should be positive"

    main(args)