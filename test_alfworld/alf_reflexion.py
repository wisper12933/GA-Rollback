import os
import sys
import json
import yaml
import torch
import importlib
import argparse
import alfworld
import alfworld.agents.environment
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from typing import Any, List, Dict, Tuple


FOLDER = '../prompts/alfworld/'
PROMPT_FILE = 'alfworld_base_prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}


with open("../prompts/alfworld/alfworld_reflexion_examples.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()


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
        generate_text = sequences[0]["generated_text"]
        if stop:
            generate_text = generate_text.split(stop)[0]

        action = extract_substring(generate_text)
        if len(action) > 1:
            action = action[0].lower() + action[1:]

        return action
        
        
class EnvironmentHistory:
    def __init__(self, base_query: str, start_info, memory: List[str], history: List[Dict[str, str]] = []) -> None:
        self._cur_query: str = f'{_get_base_query(base_query, start_info, memory)}'
        self._history: List[Dict[str, str]] = history
        self._last_action: str = ''
        self._is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation', 'human_edit']
        self._history += [{
            'label': label,
            'value': value,
        }]
        if label == 'action':
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted

    def reset(self) -> None:
        self._history = []

    def __str__(self) -> str:
        s: str = self._cur_query + '\n'
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                s += f'> {item["value"]}'
            elif item['label'] == 'observation':
                s += item['value']
            # NOT CURRENTLY SUPPORTED
            elif item['label'] == 'human_edit':
                s += f'[human edit]: {item["value"]}'
            s += '\n'
        return s


def extract_substring(s: str) -> str:
    start_pos = s.find('>')
    if start_pos > 2:
        start_pos = -1
    return s[start_pos + 1:].strip()


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
    with open(trial_log_path, 'r') as f:
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
            reflection: str = model._generate(reflection_query, '\n').strip()
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


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


def alfworld_run(env, base_prompt, memory: List[str], model, to_print=True, ob='') -> Tuple[EnvironmentHistory, bool]:
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    while cur_step < 49:
        action = model._generate(str(env_history) + ">", '\n').strip()
        env_history.add("action", action)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        env_history.add("observation", observation)
        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        if done:
            return env_history, True
        elif env_history.check_is_exhausted():
            return env_history, False
        cur_step += 1
    return env_history, False


def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        mode: str,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model,
    ) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        print(f"using {name}")

        if env_config["is_success"]:
            num_successes += 1

            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'{mode}_{v}_1'] + d[f'{mode}_{v}_0']
                final_env_history, is_success = alfworld_run(env, base_prompt, env_config["memory"] if use_memory else [], model=model, to_print=True, ob=ob)

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
                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

    # close environment object
    env.close()

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
    with open(trial_log_path, 'a') as wf:
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
    llm = Local_llm(args.local_llm_path)
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
        run_trial(trial_log_path, world_log_path, trial_idx, args.mode, env_configs, args.use_memory, llm)

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
    parser.add_argument("--num_trials", type=int, help="The number of trials to run")
    parser.add_argument("--num_envs", type=int, help="The number of environments per trial")
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--use_memory", action='store_true', help="Allow the Agent to use memory")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
    parser.add_argument("--local_llm_path", type=str, help="local path of utilized LLM")
    parser.add_argument("--mode", type=str, default='react', help="act / react")
    args = parser.parse_args()

    assert args.num_trials > 0, "Number of trials should be positive"
    assert args.num_envs > 0, "Number of environments should be positive"

    main(args)
    