import os
import json
import re
import sys
import argparse

import torch
import torch.nn.functional as F
from game24 import Game24Task
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from typing import Any, List, Dict, Tuple


file_path = '24.csv'
FOLDER = '../prompts/game24/'
ACTIONLY_FILE = 'game24_base_actionly.txt'
REACT_FILE = 'game24_base_react.txt'

with open("../prompts/game24/game24_reflexion_examples.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

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
            # ave probability
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


class IntraHistory:
    def __init__(self, base_query: str, start_info: str, memory: List[str]) -> None:
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
        s: str = self.start_info + '\n' if is_analyze else get_base_query(self.base_query, self.start_info, exp, self.memory) + '\n'
        for i in range(self.history_len):
            action = self.history['action_history'][i]
            s += f'Act {i + 1}> {action}\n'
            obs = self.history['obs_history'][i]
            s += f'Obs {i + 1}> {obs}\n'
        if len(self.history['action_history']) > self.history_len:
            s += f'Act {self.history_len + 1}> {self.last_action}\n'
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
            cur_query += '\n\nYour memory for the task below:'
            for i, m in enumerate(self.memory):
                cur_query += f'\nTrial {i}:\n{m.strip()}'
        cur_query += f"\nHere is the task:\n{self.start_info}"
        
        s: str = cur_query + '\n'
        for i in range(self.history_len):
            action = self.history['action_history'][i]
            s += f'Act {i + 1}> {action}\n'
            obs = self.history['obs_history'][i]
            s += f'Obs {i + 1}> {obs}\n'
        return s


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


def get_base_query(base_query: str, start_info: str, exp: List[str], memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour plan memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\n** Trial {i} **:\n{m.strip()}'
            
    if len(exp) > 0:
        query += '\n\nYour past experiences with the current task below:'
        for i, m in enumerate(exp):
            query += f'\n** Experience {i} **:\n{m.strip()}'
            
    query += f"\n{start_info}"
    return query


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


def format_text(text:str) -> str:
    start_pos = text.find('>')
    return text[start_pos + 1:].strip()


def generate_analysis_query(scenario: str, exp: List[str], few_shot_examples: str) -> str:
    query: str = f"""{few_shot_examples}"""

    query += f"\n# Current Task\n## Trajectory{scenario}\n##Your analysis of the current trajectory\n"
    return query


def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    return s.split("Here is the task:")[-1].strip()


def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{FEW_SHOT_EXAMPLES}

{scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nNew plan>'
    return query


def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]], model) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r', encoding='utf-8') as f:
        full_log: str = f.read()
        
    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            reflection_query: str = _generate_reflection_query(env_logs[i], memory)
            response = model._generate(reflection_query, '\n')
            reflection: str = response['text'].strip()
            env_configs[i]['memory'] += [reflection]
                
    return env_configs


def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n{start_info}"
    return query


def game24_run(env, base_prompt, memory: List[str], model, assist_model, max_rollback_num, to_print=True, ob=''):
    exp = []
    rollback_times, i = 0, 0
    
    if len(memory) > 3:
        env_history = IntraHistory(base_prompt, ob, memory[-3:])
    else:
        env_history = IntraHistory(base_prompt, ob, memory)
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()

    while True:
        query = env_history.gen_query(exp) + f'Act {env_history.history_len + 1}>'
        response = model._generate(query, stop='\n')
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
                e_loc, e_anal = gen_thought_parse(env_history, assist_model, exp)
                
                if e_loc:
                    # error detected, rollback and regenerate
                    exp.append(e_anal)
                    exp = exp[-3:]
                    env, env_history, rollback_num, action, n_query = rollback(env, env_history, model, e_loc, exp)
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
        # reward = 0 / 1
        if reward:
            return env_history, True
        elif 'Exceeded' in observation:
            return env_history, False
        if env_history.history_len > 11:
            return env_history, False


def run_trial(
        seed: int,
        num_samples: int,
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        mode: str,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model,
        assist_model,
        max_roll_num
    ) -> List[Dict[str, Any]]:
    env = Game24Task(file_path, seed, num_samples)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        curr_case = env.get_case()
        ob = '\n# Here is the task:\nInput: ' + curr_case

        if env_config["is_success"]:
            num_successes += 1

            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a', encoding='utf-8') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        if mode == 'react':
            with open(os.path.join(FOLDER, REACT_FILE), 'r') as f:
                BASE_PROMPT = f.read()
        elif mode == 'act':
            with open(os.path.join(FOLDER, ACTIONLY_FILE), 'r') as f:
                BASE_PROMPT = f.read()
        final_env_history, is_success = game24_run(env, BASE_PROMPT, env_config["memory"] if use_memory else [], model=model, assist_model=assist_model, max_rollback_num=max_roll_num, to_print=True, ob=ob)

        # update env config
        if is_success:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
            env_configs[z]['is_success'] = True
            num_successes += 1
            num_additional_successes += 1
        else:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

        # log to world log
        with open(world_log_path, 'a') as f:
            f.write(status_str + '\n')

        # log env results to trial log
        with open(trial_log_path, 'a', encoding='utf-8') as wf:
            wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 3)}
-----"""
    print(log_str)
    with open(trial_log_path, 'a', encoding='utf-8') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs


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
                'is_success': False,
                'skip': False
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
        run_trial(args.seed, args.num_envs, trial_log_path, world_log_path, trial_idx, args.mode, env_configs, args.use_memory, llm, assist_llm, args.max_roll_num)

        # update memory if needed
        if args.use_memory:
            env_configs: List[Dict[str, Any]] = update_memory(trial_log_path, env_configs, llm)

        # log env configs for trial
        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

        # log world for trial
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

        trial_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Sample seed")
    parser.add_argument("--num_trials", type=int, help="The number of trials to run")
    parser.add_argument("--num_envs", type=int, help="The number of environments per trial")
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--use_memory", action='store_true', help="Allow the Agent to use memory")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
    parser.add_argument("--max_roll_num", type=int, default=6, help="maximum number of rollbacks")
    parser.add_argument("--model_source", type=str, default="open", choices=["open", "close"])
    parser.add_argument("--llm_name_or_path", type=str, help="name of closed LLM or path of open-sourced LLM")
    parser.add_argument("--force_same_LM", type=int, default=1, help="force generator and assistant to be the same model.")
    parser.add_argument("--assist_name_or_path", type=str, default='None', help="Name or path of the LLM used as Assitant")
    parser.add_argument("--mode", type=str, default='react', help="act / react")
    args = parser.parse_args()

    assert args.num_trials > 0, "Number of trials should be positive"
    assert args.num_envs > 0, "Number of environments should be positive"

    main(args)
    