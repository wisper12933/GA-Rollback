import os
import json
import re
import sys
import openai
from openai import OpenAI
import yaml
import alfworld
import torch
import argparse
import alfworld.agents.environment
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def extract_substring(s: str) -> str:
    start_pos = s.find('>')
    return s[start_pos + 1:].strip()


def llm(prompt, stop=["\n"]):
    client = OpenAI(api_key="", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-reasoner",
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


def get_model(local_path):
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto"
    )
    
    return tokenizer, generation_pipe


def local_llm(prompt, tokenizer, generator, stop=None):
    sequences = generator(
        prompt,
        temperature=0.1,
        max_new_tokens=100,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
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


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


def alfworld_run(env, prompt, tokenizer, pipe, to_print=True, ob='', local_path=''):
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 50):
        if local_path:
            action = local_llm(init_prompt + prompt, tokenizer, pipe, stop='\n').strip()
        else:
            action = llm(init_prompt + prompt, stop=['\n']).strip()
        observation, reward, done, info = env.step([action])
        
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        if done:
            return reward
    return 0


prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_llm_path", type=str, help="local path of utilized LLM")
    parser.add_argument("--mode", type=str, help="act / react")
    args = parser.parse_args()
    
    cnts = [0] * 6
    rs = [0] * 6
    
    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    
    split = "eval_out_of_distribution"
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    
    folder = '../prompts/alfworld/'
    prompt_file = 'alfworld_base_prompts.json'
    with open(folder + prompt_file, 'r') as f:
        d = json.load(f)
        
    tokenizer, pipe = get_model(args.local_llm_path)

    for _ in range(134):
        ob, info = env.reset()
        print(f'Show: ob={ob}')
        print(f'Show: info={info}')
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(name)
        print(f'*****************************************************\nBegin Task {_+1}\n*****************************************************')
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'{args.mode}_{v}_1'] + d[f'{args.mode}_{v}_0'] + '\nHere is the task.\n'
                print(k, v)
                
                r = alfworld_run(env, prompt, tokenizer, pipe, ob=ob, local_path=args.local_llm_path)

                rs[i] += r
                cnts[i] += 1
                break
        print(_+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
        print('------------\n')
