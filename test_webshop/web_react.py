import io
import sys
import torch
import requests
import argparse
from bs4 import BeautifulSoup
from bs4.element import Comment
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


ACTION_TO_TEMPLATE = {
        'Description': 'description_page.html',
        'Features': 'features_page.html',
        'Reviews': 'review_page.html',
        'Attributes': 'attributes_page.html',
    }

prompt1 = """
Here is a Webshop task, you need to find appropriate product according to instruction and buy it (buttons
in each page are enclosed in "[]"). Below is an example:
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

prompt1_actonly = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""


def local_llm(prompt, tokenizer, generation_pipe, stop=None):
    sequences = generation_pipe(
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

    sys.stdout.flush()
    if stop:
        generate_text = generate_text.split(stop)[0]
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


def extract_substring(s: str) -> str:
    start_pos = s.find(':')
    if start_pos > 4:
        start_pos = -1
    end_pos = s.find(']', start_pos)
    if end_pos != -1:
        return s[start_pos + 1: end_pos + 1].strip()
    else:
        return s[start_pos + 1:].strip()


def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
            element.parent.name not in ignore and not isinstance(element, Comment)
    )


class webshopEnv:
    def __init__(self, website):
        self.sessions = {}
        self.website = website
        
    def webshop_text(self, session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
        if page_type == 'init':
            url = (
                f'{self.website}/{session}'
            )
        if page_type == 'search':
            url = (
                f'{self.website}/search_results/{session}/'
                f'{query_string}/{page_num}'
            )
        elif page_type == 'item':
            url = (
                f'{self.website}/item_page/{session}/'
                f'{asin}/{query_string}/{page_num}/{options}'
            )
        elif page_type == 'item_sub':
            url = (
                f'{self.website}/item_sub_page/{session}/'
                f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
            )
        elif page_type == 'end':
            url = (
                f'{self.website}/done/{session}/'
                f'{asin}/{options}'
            )
        # print(url)
        html = requests.get(url).text
        html_obj = BeautifulSoup(html, 'html.parser')
        texts = html_obj.findAll(string=True)
        visible_texts = list(filter(tag_visible, texts))
        # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
        # if page_type == 'end': import pdb; pdb.set_trace()
        if False:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            option_type = ''
            options = {}
            asins = []
            cnt = 0
            prod_cnt = 0
            just_prod = 0
            for t in visible_texts:
                if t == '\n': continue
                if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
                # if t.startswith('Instruction:') and page_type != 'init': continue
                # print(t.parent.name, t)
                if t.parent.name == 'button':  # button
                    processed_t = f'\n[{t}] '
                elif t.parent.name == 'label':  # options
                    if f"'{t}'" in url:
                        processed_t = f'[[{t}]]'
                        # observation = f'You have clicked {t}.\n' + observation
                    else:
                        processed_t = f'[{t}]'
                    options[str(t)] = option_type
                    # options[option_type] = options.get(option_type, []) + [str(t)]
                elif t.parent.get('class') == ["product-link"]:  # product asins
                    processed_t = f'\n[{t}] '
                    if prod_cnt >= 3:
                        processed_t = ''
                    prod_cnt += 1
                    asins.append(str(t))
                    just_prod = 0
                else:  # regular, unclickable text
                    processed_t = '\n' + str(t) + ' '
                    if cnt < 2 and page_type != 'init': processed_t = ''
                    if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                    option_type = str(t)
                    cnt += 1
                just_prod += 1
                observation += processed_t
            info = {}
            if options:
                info['option_types'] = options
            if asins:
                info['asins'] = asins
            if 'Your score (min 0.0, max 1.0)' in visible_texts:
                idx = visible_texts.index('Your score (min 0.0, max 1.0)')
                info['reward'] = float(visible_texts[idx + 1])
                observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
            return clean_str(observation), info

    def step(self, session, action):
        done = False
        observation_ = None
        if action == 'reset':
            self.sessions[session] = {'session': session, 'page_type': 'init'}
        elif action.startswith('think['):
            observation = 'OK.'
        elif action.startswith('search['):
            assert self.sessions[session]['page_type'] == 'init'
            query = action[7:-1]
            self.sessions[session] = {'session': session, 'page_type': 'search',
                                      'query_string': query, 'page_num': 1}
        elif action.startswith('click['):
            button = action[6:-1]
            if button == 'Buy Now':
                assert self.sessions[session]['page_type'] == 'item'
                self.sessions[session]['page_type'] = 'end'
                done = True
            elif button == 'Back to Search':
                assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
                self.sessions[session] = {'session': session, 'page_type': 'init'}
            elif button == 'Next >':
                assert False  # ad hoc page limitation
                assert self.sessions[session]['page_type'] == 'search'
                self.sessions[session]['page_num'] += 1
            elif button == '< Prev':
                assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
                if self.sessions[session]['page_type'] == 'search':
                    assert False
                    self.sessions[session]['page_num'] -= 1
                elif self.sessions[session]['page_type'] == 'item_sub':
                    self.sessions[session]['page_type'] = 'item'
                elif self.sessions[session]['page_type'] == 'item':
                    self.sessions[session]['page_type'] = 'search'
                    self.sessions[session]['options'] = {}
            elif button in ACTION_TO_TEMPLATE:
                assert self.sessions[session]['page_type'] == 'item'
                self.sessions[session]['page_type'] = 'item_sub'
                self.sessions[session]['subpage'] = button
            else:
                if self.sessions[session]['page_type'] == 'search':
                    assert button in self.sessions[session].get('asins', [])  # must be asins
                    self.sessions[session]['page_type'] = 'item'
                    self.sessions[session]['asin'] = button
                elif self.sessions[session]['page_type'] == 'item':
                    assert 'option_types' in self.sessions[session]
                    assert button in self.sessions[session]['option_types'], (
                        button, self.sessions[session]['option_types'])  # must be options
                    option_type = self.sessions[session]['option_types'][button]
                    if not 'options' in self.sessions[session]:
                        self.sessions[session]['options'] = {}
                    self.sessions[session]['options'][option_type] = button
                    observation_ = f'You have clicked {button}.'
        else:
            assert False
        observation, info = self.webshop_text(**self.sessions[session])
        if observation_:
            observation = observation_
        self.sessions[session].update(info)
        reward = info.get('reward', 0.0)
        return observation, reward, done


def webshop_run(idx, prompt, env, tokenizer, pipe, to_print=True):
    action = 'reset'
    init_prompt = prompt
    prompt = ''
    for i in range(15):
        try:
            res = env.step(idx, action)
            observation = res[0]
        except AssertionError:
            observation = 'Invalid action!'

        if action.startswith('think'):
            observation = 'OK.'

        if to_print:
            print(f'Action{i+1}: {action}\nObservation{i+1}: {observation}\n')
            sys.stdout.flush()
        if i:
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
        else:
            prompt += f'{observation}\n\nAction:'

        if res[2]:
            return res[1]

        action = local_llm(init_prompt + prompt[-(6400 - len(init_prompt)):], tokenizer, pipe, stop='\n').lstrip(' ')

    return 0


def run_episodes(prompt, env, tokenizer, pipe, n=50):
    rs = []
    cnt = 0
    for i in range(n):
        print('---------------------------------------------------------------------------')
        sys.stdout.flush()
        try:
            r = webshop_run(f'fixed_{i}', prompt, env, tokenizer, pipe, to_print=True)
        except AssertionError:
            r = 0
            cnt += 1
        rs.append(r)
        if (i + 1) % 1 == 0:
            r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
            print(f"id:{i + 1}, score:{r}, success rate:{sr}, false rate{fr}")
            print('---------------------------------------------------------------------------')
            sys.stdout.flush()
    r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
    print(f"score:{r}, success rate:{sr}, false rate{fr}")
    sys.stdout.flush()
    return rs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, help="webshop_url")
    parser.add_argument("--local_llm_path", type=str, help="local path of utilized LLM")
    parser.add_argument("--mode", type=str, help="act / react")
    args = parser.parse_args()
    
    env = webshopEnv(args.website)
    tokenizer, pipe = get_model(args.local_llm_path)

    if args.mode == 'react':
        res1 = run_episodes(prompt1, env, tokenizer, pipe, 500)
    elif args.mode == 'act':
        res1 = run_episodes(prompt1_actonly, env, tokenizer, pipe, 500)
    else:
        print('Wrong Mode')
