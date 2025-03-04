import re
import os
import sympy
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Tuple


def get_base_query(base_query: str, start_info: str, exp: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(exp) > 0:
        query += '\n\nYour past experiences with the current task below:'
        for i, m in enumerate(exp):
            query += f'\n** Experience {i} **:\n{m.strip()}'
    query += f"\n{start_info}"
    return query


class Game24Task():
    """
    Case: a string of 4 numbers
    Case Example: 
        1 2 3 4
    Trajectory Example: 
        Act 1> 1 + 2 = 3 
        Obs 1> numbers left: 3 3 4
        Act 2> 3 + 3 = 6
        Obs 2> numbers left: 4 6
        Act 3> 6 * 4 = 24
        Obs 3> numbers left: 24
        Act 4> answer: (1 + 2 + 3) * 4 = 24
        Obs 4> Success!
    """
    def __init__(self, file_path, seed, sample_num):
        self.data = pd.read_csv(file_path)['Puzzles']
        self.seed = seed
        self.sample_num = sample_num
        self.sampled_data = self.shuffle_and_sample()
        
        self.idx = -1
        self.max_steps = 4
        self.step_cnt = 0
        self.value_cache = []
    
    def shuffle_and_sample(self):
        np.random.seed(self.seed)
        shuffled_data = self.data.sample(frac=1, random_state=self.seed)
        return shuffled_data.head(self.sample_num)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def step(self, action: str):
        if action.startswith('think'):
            return {'r': 0, 'ob': 'OK.'}

        self.step_cnt += 1
        if (self.step_cnt > self.max_steps):
            return {'r': 0, 'ob': 'Exceeded the allowed number of steps.'}
        
        if action.startswith('answer'):
            expression = action.replace('answer: ', '').split('=')[0]
            numbers = re.findall(r'\d+', expression)
            problem_numbers = re.findall(r'\d+', self.data[self.idx])
            if sorted(numbers) != sorted(problem_numbers):
                return {'r': 0, 'ob': 'The numbers in the math expression do not match the numbers in the problem!'}
            try:
                reward = int(sympy.simplify(expression) == 24)
                return {'r': reward, 'ob': 'Success!' if reward else 'The result of the expression is not 24.'}
            except Exception as e:
                return {'r': 0, 'ob': str(e)}
        elif self.step_cnt == self.max_steps:
            self.step_cnt -= 1
            return {'r': 0, 'ob': 'All available numbers have been consumed. Please provide the final arithmetic expression, which should start with "answer: ", such as: “answer: (6 - 4) * (4 + 8) = 24”'}
        else:
            try:
                split_expression = action.split('=')
                if len(split_expression) == 2:
                    consumed_number = re.findall(r'\d+', split_expression[0])
                    gen_number = re.findall(r'\d+', split_expression[1])
                    
                    cache_cnt = self.value_cache.copy()
                    for num in consumed_number:
                        if num in cache_cnt:
                            cache_cnt.remove(num)
                    cache_cnt.extend(gen_number)
                    self.value_cache = sorted(cache_cnt)
                    
                    return {'r': 0, 'ob': 'numbers left: ' + ' '.join(self.value_cache)}
                else:
                    return {'r': 0, 'ob': 'Invalid Action'}
            except Exception as e:
                return {'r': 0, 'ob': str(e)}
    
    def get_case(self, reset=False):
        self.step_cnt = 0
        if not reset:
            self.idx += 1
        task_case = self.data[self.idx]
        self.value_cache = re.findall(r'\d+', task_case)
        return task_case


class IntraHistory():
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
    