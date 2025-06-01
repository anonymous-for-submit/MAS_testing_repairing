import os
import copy
import json
import argparse
import tqdm
import random
# from session import Session

import copy
import time
# from main_fuzz_passAt10 import PromptNode
class PromptNode:
    def __init__(self,
                 solution,
                 score=0,
                 passes=0,
                 parent: 'PromptNode' = None):

        self.solution = solution


        self.visited_num = 0
        self.score=score
        self.passes=passes
        self.reward_score=0
        self.finish = False
        

        self.parent: 'PromptNode' = parent
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

# EXPAND_1_SENTANCE='''
# I will give you a coding question prompt, with several test cases. You are required to add a sentence to the end of the description part of the question template, and return the whole question. Do not make any other explanation nor have beginning or ending indicator in your answer. 
# Here is the question:
# '''

ADD_1_SENTANCE_AT_END='''
I will give you a coding question prompt, with several test cases. You are required to add a sentence to the end of the description part of the question template, and return the whole question. Do not make any change to the other part of the question. Do not make any change to the meaning of the question. You should not change the input format and output format. Do not make any other explanation nor have beginning or ending indicator in your answer. 
RETURN THE COMPLETED QUESTION!
Here is the question:
'''


EXPAND_ALL='''
I will give you a coding question prompt, with several test cases. There are natural language description between code method name and test cases. You are required to expand the natural language description part of the question template. Do not make any change to the code and test cases. Do not make any change to the meaning of the question. Do not make any other explanation nor have beginning or ending indicator in your answer. 
YOU CAN ONLY EXPAND THE NATURAL LANGUAGE PART, DO NOT MAKE ANY CHANGE TO OTHER PART!
Here is the question:
'''

SHORTEN='''
I will give you a coding question prompt, with several test cases. You are required to condense sentences you think are too long and delete the meaningless sentence. Also, you should maintain the overall meaning of the template and SHOULD NOT delete the test cases in the templete. Do not make any change to the meaning of the question. You should not change the input format and output format. Do not make any other explanation nor have beginning or ending indicator in your answer. 
Here is the question:
'''

REPHRASE='''
I will give you a coding question prompt, with several test cases. You are required to rephrase sentences in the natural language description part while remaining other sentences unchanged. Also, you should maintain the overall meaning of the template and SHOULD NOT delete the test cases. Do not make any change to the meaning of the question. You should not change the input format and output format. Do not make any other explanation nor have beginning or ending indicator in your answer. 
Here is the question:
'''

CHANGE_IDENTIFIER_FUNCNAME = '''
I will give you a coding question prompt, with several test cases. You are required to change the identifier of the given code into random strings, while remaining other sentences unchanged. Do not change the function name! Also, you should maintain the overall meaning of the template and SHOULD NOT delete the test cases. Do not make any change to the meaning of the question. You should not change the input format and output format. Do not make any other explanation nor have beginning or ending indicator in your answer. 
Here is the question:
'''


CONDENSE_ONE_SENTENCE='''
I will give you a coding question prompt, with several test cases. You are required to randomly choose one sentence from the question description, condense the sentence and delete useless information in the sentence. Do not make any change to other sentences.
Also, you should maintain the overall meaning of the question.
You SHOULD NOT delete the test cases or before function in the templete!! 
Do not make any change to the meaning of the question. You should not change the input format and output format. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Only return the whole question after your mutation.
Here is the question:
'''

CONDENSE_TWO_SENTENCE_INTO_ONE = '''
I will give you a coding question prompt, with several test cases. You are required to randomly choose two consecutive sentences from the question description and condense them into one sentence. Do not make any change to other sentences. If there is only one sentence in the question description, do not make any change to it.
Also, you should maintain the overall meaning of the question.
You SHOULD NOT delete the test cases or before function in the templete!! 
Do not make any change to the meaning of the question. You should not change the input format and output format. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Only return the whole question after your mutation.
Here is the question:
'''

EXPAND_ONE_SENTENCE_INTO_TWO = '''
I will give you a coding question prompt, with several test cases. You are required to randomly choose one sentence from the question description and expand it into two sentences. Do not make any change to other sentences. 
Also, you should maintain the overall meaning of the question.
You SHOULD NOT delete the test cases or before function in the templete!! 
Do not make any change to the meaning of the question. You should not change the input format and output format. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Only return the whole question after your mutation.
Here is the question:
'''

EXPAND_ONE_SENTENCE = '''
I will give you a coding question prompt, with several test cases. You are required to randomly choose ONE sentence from the question description, add more useful information to the sentence. Do not make any change to other sentences.
Also, you should maintain the overall meaning of the question.
You SHOULD NOT delete the test cases or before function in the templete!! 
Do not make any change to the meaning of the question. You should not change the input format and output format. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Only return the whole question after your mutation.
Here is the question:
'''

REPHRASE_ONE_SENTENCE = '''
I will give you a coding question prompt, with several test cases. You are required to randomly choose ONE sentence from the question description, and use other words to rewrite the sentence. Do not make any change to other sentences.
Also, you should maintain the overall meaning of the question.
You SHOULD NOT delete the test cases or before function in the templete!! 
Do not make any change to the meaning of the question. You should not change the input format and output format. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Only return the whole question after your mutation.
Here is the question:
'''

NL_ADD_1_SENTANCE_AT_END='''
I will give you a coding question prompt. You are required to add a sentence to the end of the description part of the question template, and return the whole question. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Return the whole question after your mutation!
Here is the question:
'''


NL_EXPAND_ALL='''
I will give you a coding question prompt. There are natural language description between code method name and test cases. You are required to expand the natural language description part of the question template.
Do not make any change to the meaning of the question. Do not make any other explanation nor have beginning or ending indicator in your answer. 
Return the whole question after your mutation!
Here is the question:
'''

NL_SHORTEN='''
I will give you a coding question prompt. You are required to condense sentences you think are too long and delete the meaningless sentence. Also, you should maintain the overall meaning of the question. 
Do not make any other explanation nor have beginning or ending indicator in your answer.
Return the whole question after your mutation! 
Here is the question:
'''

NL_REPHRASE='''
I will give you a coding question prompt. You are required to rephrase the question while maintaining the overall meaning. Do not make any change to the meaning of the question. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Return the whole question after your mutation!
Here is the question:
'''


NL_CONDENSE_ONE_SENTENCE='''
I will give you a coding question prompt. You are required to randomly choose one sentence from the question description, condense the sentence and delete useless information in the sentence. Do not make any change to other sentences.
Also, you should maintain the overall meaning of the question.
Do not make any change to the meaning of the question. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Return the whole question after your mutation!
Here is the question:
'''

NL_CONDENSE_TWO_SENTENCE_INTO_ONE = '''
I will give you a coding question prompt. You are required to randomly choose two consecutive sentences from the question description and condense them into one sentence. Do not make any change to other sentences. If there is only one sentence in the question description, do not make any change to it.
Also, you should maintain the overall meaning of the question.
Do not make any change to the meaning of the question. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Return the whole question after your mutation!
Here is the question:
'''

NL_EXPAND_ONE_SENTENCE_INTO_TWO = '''
I will give you a coding question prompt. You are required to randomly choose one sentence from the question description and expand it into two sentences. Do not make any change to other sentences. 
Also, you should maintain the overall meaning of the question.
Do not make any change to the meaning of the question. 
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Return the whole question after your mutation!
Here is the question:
'''

NL_EXPAND_ONE_SENTENCE = '''
I will give you a coding question prompt. You are required to randomly choose ONE sentence from the question description, add more useful information to the sentence. Do not make any change to other sentences.
Also, you should maintain the overall meaning of the question.
Do not make any change to the meaning of the question. 
Do not make any other explanation nor have beginning or ending indicator in your answer.
Return the whole question after your mutation!
Here is the question:
'''

NL_REPHRASE_ONE_SENTENCE = '''
I will give you a coding question prompt. You are required to randomly choose ONE sentence from the question description, and use other words to rewrite the sentence. Do not make any change to other sentences.
Also, you should maintain the overall meaning of the question.
Do not make any change to the meaning of the question. 
Return the whole question after your mutation!
Do not make any other explanation nor have beginning or ending indicator in your answer. 
Here is the question:
'''



import openai
from openai import OpenAI
from openai import AzureOpenAI
# client = OpenAI(
#     # 输入转发API Key
#     api_key="sk-NsLLS6Bbm06SDgbx3BJkyHsEys50pj9TqlZB7PrIJHFSIzmI",
#     base_url="https://api.chatanywhere.com.cn/v1"
# )

client = AzureOpenAI(
        azure_endpoint = "https://hkust.azure-api.net", 
        api_key="f9f10057a7e749898daeabdf5f6b84be",  
        api_version="2024-02-01"
    )
def call_deepseek_coder(prompt, model='deepseek-coder', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, echo=False, majority_at=None):
    if model == 'deepseek-coder':
        model = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    openai_api_key = "EMPTY"
    openai_api_base = "http://127.0.0.1:8000/v1/"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    # print('in ds coder!')
    # print('prompt')
    # print(prompt)

    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10

    completions = []
    for i in range(3):
        try:
            # print('***'*30)
            # print(prompt)
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            # print(client.api_key)
            # print(client.base_url)
            # print(max_tokens,temperature,top_p,requested_completions)
            response = client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=requested_completions
            )
            while not response:
                response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions
                    )
            completions.extend([choice.message.content for choice in response.choices])
            # print(completions[0])
            # print('*'*30)
            if len(completions) >= num_completions:
                # print('completion')
                # print(completions[0])
                return completions[:num_completions]
        except Exception as e:
            time.sleep(min(i**2, 60))
            print(e)
            print(prompt)
    raise RuntimeError('Failed to call GPT API')
# def mutate_ds(model,prompt,mutate_prompt):
#     prompt=mutate_prompt+prompt
#     message = [
#         {"role": "user", "content": prompt}
#     ]
#     completions = []
#     for _ in range(5):
#         try:      
#             requested_completions=1
#             max_tokens=512
#             temperature=1
#             num_completions=1
#             response = ds_client.chat.completions.create(
#             model=model,
#             messages=message,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             n=requested_completions
#             )
#             completions.extend([choice.message.content for choice in response.choices])
#             if len(completions) >= num_completions:
#                 return completions[:num_completions]
#         except Exception:
#             time.sleep(20)
#     print('Failed to call GPT API')
#     return ['']
#     # raise RuntimeError('Failed to call GPT API')
def mutate(model,prompt,mutate_prompt):
    prompt=mutate_prompt+prompt
    message = [
        {"role": "user", "content": prompt}
    ]
    completions = []
    if 'deepseek' in model:
        return call_deepseek_coder(message,model)
    for _ in range(5):
        try:      
            requested_completions=1
            max_tokens=512
            temperature=1
            num_completions=1
            response = client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature,
            n=requested_completions
            )
            completions.extend([choice.message.content for choice in response.choices])
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except Exception as e:
            print(e)
            time.sleep(20)
        
    print('Failed to call GPT API')
    return ['']
    raise RuntimeError('Failed to call GPT API')


mutate_prompt_map = {'add_1_sentence_at_end':ADD_1_SENTANCE_AT_END,'rephrase':REPHRASE,'shorten':SHORTEN, 'expand_one':EXPAND_ONE_SENTENCE,'condense_one':CONDENSE_ONE_SENTENCE,'expand_one2two':EXPAND_ONE_SENTENCE_INTO_TWO,'condense_two2one':CONDENSE_TWO_SENTENCE_INTO_ONE,'rephrase_one':REPHRASE_ONE_SENTENCE}
mutate_prompt_nl_map = {'add_1_sentence_at_end':NL_ADD_1_SENTANCE_AT_END,'rephrase':NL_REPHRASE,'shorten':NL_SHORTEN, 'expand_one':NL_EXPAND_ONE_SENTENCE,'condense_one':NL_CONDENSE_ONE_SENTENCE,'expand_one2two':NL_EXPAND_ONE_SENTENCE_INTO_TWO,'condense_two2one':NL_CONDENSE_TWO_SENTENCE_INTO_ONE,'rephrase_one':NL_REPHRASE_ONE_SENTENCE}

mutate_methods=['add_1_sentence_at_end','rephrase','shorten','expand_one','condense_one','expand_one2two','condense_two2one','rephrase_one']
# mutate_methods = mutate_methods[3:]


def mutate_one(seed,args,mutate_method='random',model='gpt-4o'):
    mutate_methods=['add_1_sentence_at_end','rephrase','shorten','expand_one','condense_one','expand_one2two','condense_two2one','rephrase_one']
    intent = seed.solution['prompt']
    # print(intent)
    mutated_prompt=''
    if mutate_method == 'random':
        if args.mutate_level == 'whole':
            mutate_methods = mutate_methods[:3]
        elif args.mutate_level == 'sentence':
            # print(11111)
            mutate_methods = mutate_methods[3:]
        mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]

    if mutate_method not in mutate_prompt_map.keys():
        print('not implemented')
        raise NotImplementedError

    prompt4mutation = mutate_prompt_map[mutate_method]
    
    mutated_prompt = mutate(model,intent,prompt4mutation)[0]

    if not mutated_prompt:
        mutated_prompt = intent

    # if 'MBPP' not in args.dataset:
    #     while 'def ' not in mutated_prompt:
    #         print('改变了prompt的结构!!!')
    #         mutated_prompt = mutate(model,intent,prompt4mutation)[0]

    new_solution=copy.deepcopy(seed.solution)
    new_solution['prompt'] = mutated_prompt
    print('-'*100)
    print(intent)
    print('-'*100)
    print(mutated_prompt)
    print('-'*100)
    ans=PromptNode(solution=new_solution,parent=seed)
    return ans,mutate_method


def mutate_one_nl(seed,args,mutate_method='random',model='gpt-4o'):
    if args.clean_mutate_method==1:
        mutate_methods=['add_1_sentence_at_end','rephrase','shorten','expand_one2two','condense_two2one','rephrase_one','add_1_sentence_at_end']
    else:
        mutate_methods=['add_1_sentence_at_end','rephrase','shorten','expand_one','condense_one','expand_one2two','condense_two2one','rephrase_one']
    
    if 'human' in args.dataset:
        intent = seed.solution['nl']
    elif 'contest' in args.dataset:
        examples_idx = seed.solution['prompt'].find('\nInput\n')
        intent = seed.solution['prompt'][:examples_idx]
        seed.solution['examples'] = seed.solution['prompt'][examples_idx:]
        
    else:
        intent = seed.solution['prompt']
        
    # print(intent)
    mutated_nl=''
    if args.mutate_level == 'whole':
        mutate_methods = mutate_methods[:3]
    elif args.mutate_level == 'sentence':
        # print(11111)
        mutate_methods = mutate_methods[3:]

    # print(mutate_methods)
    if 'wo_' in mutate_method:
        mutate_method_wo = mutate_method[4:]
        mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]
        while mutate_method == mutate_method_wo:
            mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]
    if mutate_method == 'random':
        mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]
    
         
    if mutate_method not in mutate_prompt_nl_map.keys():
        print('not implemented')
        raise NotImplementedError

    prompt4mutation = mutate_prompt_nl_map[mutate_method]
    
    mutated_nl = mutate(model,intent,prompt4mutation)[0]


    new_solution=copy.deepcopy(seed.solution)
    if 'human' in args.dataset:
        new_solution['prompt'] = seed.solution['func']+'\t\n\'\'\''+mutated_nl+'\n'+seed.solution['examples'] +'\'\'\''
    elif 'contest' in args.dataset:
        new_solution['prompt'] = mutated_nl + '\n' + new_solution['examples']
    else:
        new_solution['prompt'] = mutated_nl
    
    print('-'*50)
    print(mutate_method)
    print('-'*50)
    print(intent)
    print('-'*50)
    print(new_solution['prompt'])
    print('-'*50)

    ans=PromptNode(solution=new_solution,parent=seed)
    return ans,mutate_method

def get_more_prompt(seed,args,mutate_method='random',model='gpt-4o'):
    if args.clean_mutate_method==1:
        mutate_methods=['add_1_sentence_at_end','rephrase','shorten','expand_one2two','condense_two2one','rephrase_one','add_1_sentence_at_end']
    else:
        mutate_methods=['add_1_sentence_at_end','rephrase','shorten','expand_one','condense_one','expand_one2two','condense_two2one','rephrase_one']
    
    intent = seed.solution['nl']
    # print(intent)
    mutated_nl=''
    if args.mutate_level == 'whole':
        mutate_methods = mutate_methods[:3]
    elif args.mutate_level == 'sentence':
        # print(11111)
        mutate_methods = mutate_methods[3:]

    # print(mutate_methods)
    if 'wo_' in mutate_method:
        mutate_method_wo = mutate_method[4:]
        mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]
        while mutate_method == mutate_method_wo:
            mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]
    if mutate_method == 'random':
        mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]
    
         
    if mutate_method not in mutate_prompt_nl_map.keys():
        print('not implemented')
        raise NotImplementedError

    prompt4mutation = mutate_prompt_nl_map[mutate_method]
    
    mutated_nl = mutate(model,intent,prompt4mutation)[0]

    if not mutated_nl:
        mutated_nl = intent

    new_prompt = seed.solution['func']+'\t\n\'\'\''+mutated_nl+'\n'+seed.solution['examples'] +'\'\'\''
    
    return new_prompt

def get_more_prompt_base_dataset(task,args,mutate_method='random',model='gpt-4o'):
    if args.clean_mutate_method==1:
        mutate_methods=['add_1_sentence_at_end','rephrase','shorten','expand_one2two','condense_two2one','rephrase_one','add_1_sentence_at_end']
    else:
        mutate_methods=['add_1_sentence_at_end','rephrase','shorten','expand_one','condense_one','expand_one2two','condense_two2one','rephrase_one']
    
    intent = task['nl']
    # print(intent)
    mutated_nl=''
    mutate_methods = mutate_methods[3:]

    # print(mutate_methods)
    if 'wo_' in mutate_method:
        mutate_method_wo = mutate_method[4:]
        mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]
        while mutate_method == mutate_method_wo:
            mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]
    if mutate_method == 'random':
        mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]
    
         
    if mutate_method not in mutate_prompt_nl_map.keys():
        print('not implemented')
        raise NotImplementedError

    prompt4mutation = mutate_prompt_nl_map[mutate_method]
    
    mutated_nl = mutate(model,intent,prompt4mutation)[0]

    if not mutated_nl:
        mutated_nl = intent

    new_prompt = task['func']+'\t\n\'\'\''+mutated_nl+'\n'+task['examples'] +'\'\'\''
    
    return new_prompt

def get_more_prompt_test(prompt,args,mutate_method='random'):
    mutate_methods=['add_1_sentence_at_end','rephrase','shorten','expand_one2two','condense_two2one','rephrase_one','add_1_sentence_at_end']
    intent = prompt

    split_examples =False
    examples=''
    if '\nInput\n' in prompt:
        examples_idx = prompt.find('\nInput\n')
        intent = prompt[:examples_idx]
        examples = prompt[examples_idx:]
        split_examples=True
    # print(intent)
    # print('$$$'*39)
    # print(examples)


    # print(intent)
    mutated_prompt=''
    if mutate_method == 'random':
        mutate_methods = mutate_methods[3:]
        mutate_method = mutate_methods[random.randint(0,len(mutate_methods)-1)]

    if 'human' in args.dataset or 'Human'  in args.dataset:
        prompt_map = mutate_prompt_map
    if 'mbpp' in args.dataset or 'MBPP' in args.dataset:
        prompt_map = mutate_prompt_nl_map
    if 'contest' in args.dataset:
        prompt_map = mutate_prompt_nl_map

    if mutate_method not in prompt_map.keys():
        print('not implemented')
        raise NotImplementedError

    prompt4mutation = prompt_map[mutate_method]
    
    mutated_prompt = mutate(args.model,intent,prompt4mutation)[0]

    if not mutated_prompt:
        mutated_prompt = intent

    if split_examples:
        mutated_prompt+='\n'+examples
    # print(mutate_method)
    # print(mutated_prompt)

    # while 'def ' not in mutated_prompt:
    #     print('改变了prompt的结构!!!')
    #     mutated_prompt = mutate(model,intent,prompt4mutation)[0]
    
    return mutated_prompt
      
CHOOSE_PROMPT='''
I would give you a list of {} solutions for the input coding question. Please Choose one solution among these solutions that you think is the correct one, and return the index of code.
ONLY return one integer and do not make explaination

coding question:
<\question>
solution code
<\solutions>
'''
def choose_code(codes,prompt,model='gpt-4o'):
    solutions=''
    for i in range(len(codes)):
        solutions+='code {}:\n'.format(i)
        solutions+=codes[i]
    prompt=CHOOSE_PROMPT.replace('<\question>',prompt).replace('<\solutions>',solutions)
    # print('#'*20+'in choosing code'+'#'*20)
    # print(prompt)
    message = [
        {"role": "user", "content": prompt}
    ]
    completions = []
    for _ in range(20):
        try:      
            requested_completions=1
            max_tokens=512
            temperature=0
            num_completions=1
            response = client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature,
            n=requested_completions
            )
            completions.extend([choice.message.content for choice in response.choices])
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except Exception:
            time.sleep(20)
    raise RuntimeError('Failed to call GPT API')
'''
3. For each step of the plan, please provide sample code to implement this step.
Please noted that you should only provide the code for that step!
4. For each test case in the requirement, analysis the input and output, and explain the logic step by step from input to output.

6. Test case analysis:
   - Test case 1: fib(10)
     Input: n = 10
     Output: 55
     Logic: Calculate the 10th Fibonacci number, which is fib(9) + fib(8) = 55.

   - Test case 2: fib(1)
     Input: n = 1
     Output: 1
     Logic: Calculate the 1st Fibonacci number, which is 1.

   - Test case 3: fib(8)
     Input: n = 8
     Output: 21
     Logic: Calculate the 8th Fibonacci number, which is 21.
'''


REPAIR_PROMPT_W_CORE_COMCEPT = '''
Here is a coding requirement from the user and plan from analyst
requirement:
<r>
plan
<p>
I want you to read the plan and requirement, and provide some insight for the coder based on the follow perspective.
Noted that you should answer precise and correct.
1. Identify the core concept(key words, important concept) of the requirement, and explain the meaning of core concept.
2. Identify all the phrase showing quantity relationship (greater than, more than, two times, two multiply two, as much as) in the requirement, and explain the meaning of them in the requirement,then show how to implement them in code.
3. Identify all degree adverb (largest, greatest, best, shorest) in the requirement, and explain the meaning of them  in the requirement, then show how to implement them in code.
4. For the steps in plan, check if some steps should be implement simultaneously (in one code block or if-else statement), and explain the implementation
5. Based on the requirement and analysis, identify the edge case of the question, generate three edge case based on the format of edge cases in the requirement, and identify the correct output of edge case and explain it.
6. Based on the requirement and analysis, Do we need extra code to handle the edge cases, or it could be solved in original code? If extra code is needed, please write sample code to handle the edge case. PLEASE ONLY WRITE THE CODE HANDLING THE EDGE CASE!

- The format of test cases should be:
1. core concept: <core concept>
   Explanation: ...
1. <phrase>: <explanation> 
   ...
2. <degree adverb>: <explanation> 
   ... 
4. (check if there are steps should be considered simultaneously)
5. <edge case> = <expected output>
   Explanation:
   ...
6. We need extra code to handle the edge cases.
    (code for handling the edge case)

# For example:
## Prompt 1:
requirement:
def how_many_times(string: str, substring: str) -> int:
\'\'\' Find how many times a specific substring appears within the original string. Include overlapping instances.
>>> how_many_times('', 'a')
    0
    >>> how_many_times('aaa', 'a')
    3
    >>> how_many_times('aaaa', 'aa')
    3
    \'\'\'
plan
{
  "plan": {
    "subproblems": [
      "Identify the length of the original string",
      "Identify the length of the substring",
      "Iterate through the original string to find all occurrences of the substring",
      "Count the number of occurrences found"
    ],
    "steps": [
      "Get the input string and substring from the user",
      "Initialize a counter variable to keep track of the number of occurrences",
      "Iterate through the original string using a sliding window approach",
      "Check if the current substring matches the input substring",
      "If a match is found, increment the counter variable",
      "Return the final count of occurrences"
    ]
  }
}

## Completion 1:
1. core concept: overlapping
   In the requirement it means that we could count the overlapping apperance of substring in the original string

2. No phrase showing quantity relationship

3. No degree adverb

4. The step 3-5 should be implement simultaneously
    "Iterate through the original string using a sliding window approach",
    "Check if the current substring matches the input substring",
    "If a match is found, increment the counter variable"
    This could be done by writing one for loop to iterate through the orginal string, extract every substring with the size of substring, check if it match the input substring and increment the counter variable if a match is found

5. how_many_times('', 'a') = 0
   explanation: Since the original string is empty, the substring cannot appear, so the expected output is 0.
   how_many_times('abc', '') = 4
   explanation: '' appears four times in the orginal string. 'abc'.count('')=2

6. Extra code are needed to handle the edge case.
    if not string:
        return 0
    elif not substring:
        return len(string)+1
    (other code)


## Prompt 2:

requirement:
def search(lst):	
\'\'\'You are given a non-empty list of positive integers. Return the largest integer that is more than zero and appears at least as many times as the integer itself. If no such a value exist, return -1.
        search([4, 1, 2, 2, 3, 1]) == 2
        search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3
        search([5, 5, 4, 4, 4]) == -1
    \'\'\'
plan:
{
  "plan": {
    "subproblems": [
      "Identify the frequency of each integer in the list",
      "Find the largest integer that appears at least as many times as itself",
      "Handle the case where no such integer exists"
    ],
    "steps": [
      "Create a dictionary to store the frequency of each integer in the list",
      "Iterate through the list and update the frequency in the dictionary",
      "Iterate through the dictionary to find the largest integer that meets the condition",
      "Return the result or -1 if no such integer exists"
    ]
  }
}
}

## Completion 2:
1. core concept: positive, at least as many times
   positive means that all interger in the list is > 0
   at least as many times means appears of a number >= its value

2. 'more than': means that we need to find interger > 0
   'at least as many times': means that we need to find the interger whose appears times is greater than or equal to its value

3. 'largest': means that we need the bigest interger that appears greater or equal to its value

4. There are no steps that could be implement simultaneously. All 4 steps shoule be implement step by step.

5. search([2,2,3,3,3]) = 3
   explanation: Both 2 and 3 appears greater than or equal to its value, but 3 is the largest number
   search([3,3,2,4,4,4]) = -1
   explanation: number 2 appears one time, number 3 appears two times,number 4 appears three times, none of them appears greater than or equal to its value, so the function return -1

6. We do not need extra code to handle the edge case. We could set the original return answer to -1 and then find the largest interger that meets the need. 

## Prompt 3:
requirement:
<r>
plan
<p>

## Answer 3:
'''
REPAIR_PROMPT= '''
Here is a coding requirement from the user and plan from analyst
requirement:
<r>
plan
<p>
I want you to read the plan and requirement, and provide some insight for the coder based on the follow perspective.
Noted that you should answer precise and correct.

1. Identify all the phrase showing quantity relationship (greater than, more than, two times, two multiply two, as much as) in the requirement, and explain the meaning of them in the requirement,then show how to implement them in code.
2. Identify all degree adverb (largest, greatest, best, shorest) in the requirement, and explain the meaning of them in the requirement, then show how to implement them in code.
3. For the steps in plan, check if some steps should be implement simultaneously (in one code block or if-else statement), and explain the implementation
4. Based on the requirement and analysis, identify the edge case of the question, generate three edge case based on the format of edge cases in the requirement, and identify the correct output of edge case and explain it.
5. Based on the requirement and analysis, Do we need extra code to handle the edge cases, or it could be solved in original code? If extra code is needed, please write sample code to handle the edge case. PLEASE ONLY WRITE THE CODE HANDLING THE EDGE CASE!

- The format of test cases should be:
1. <phrase>: <explanation> 
   ...
2. <degree adverb>: <explanation> 
   ... 
3. (check if there are steps should be considered simultaneously)
4. <edge case> = <expected output>
   Explanation:
   ...
5. We need extra code to handle the edge cases.
    (code for handling the edge case)

# For example:
## Prompt 1:
requirement:
def how_many_times(string: str, substring: str) -> int:
\'\'\' Find how many times a specific substring appears within the original string. Include overlapping instances.
>>> how_many_times('', 'a')
    0
    >>> how_many_times('aaa', 'a')
    3
    >>> how_many_times('aaaa', 'aa')
    3
    \'\'\'
plan
{
  "plan": {
    "subproblems": [
      "Identify the length of the original string",
      "Identify the length of the substring",
      "Iterate through the original string to find all occurrences of the substring",
      "Count the number of occurrences found"
    ],
    "steps": [
      "Get the input string and substring from the user",
      "Initialize a counter variable to keep track of the number of occurrences",
      "Iterate through the original string using a sliding window approach",
      "Check if the current substring matches the input substring",
      "If a match is found, increment the counter variable",
      "Return the final count of occurrences"
    ]
  }
}

## Completion 1:
1. No phrase showing quantity relationship

2. No degree adverb

3. The step 3-5 should be implement simultaneously
    "Iterate through the original string using a sliding window approach",
    "Check if the current substring matches the input substring",
    "If a match is found, increment the counter variable"
    This could be done by writing one for loop to iterate through the orginal string, extract every substring with the size of substring, check if it match the input substring and increment the counter variable if a match is found

4. how_many_times('', 'a') = 0
   explanation: Since the original string is empty, the substring cannot appear, so the expected output is 0.
   how_many_times('abc', '') = 4
   explanation: '' appears four times in the orginal string. 'abc'.count('')=2

5. Extra code are needed to handle the edge case.
    if not string:
        return 0
    elif not substring:
        return len(string)+1
    (other code)


## Prompt 2:

requirement:
def search(lst):	
\'\'\'You are given a non-empty list of positive integers. Return the largest integer that is more than zero and appears at least as many times as the integer itself. If no such a value exist, return -1.
        search([4, 1, 2, 2, 3, 1]) == 2
        search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3
        search([5, 5, 4, 4, 4]) == -1
    \'\'\'
plan:
{
  "plan": {
    "subproblems": [
      "Identify the frequency of each integer in the list",
      "Find the largest integer that appears at least as many times as itself",
      "Handle the case where no such integer exists"
    ],
    "steps": [
      "Create a dictionary to store the frequency of each integer in the list",
      "Iterate through the list and update the frequency in the dictionary",
      "Iterate through the dictionary to find the largest integer that meets the condition",
      "Return the result or -1 if no such integer exists"
    ]
  }
}
}

## Completion 2:
1. 'more than': means that we need to find interger > 0
   'at least as many times': means that we need to find the interger whose appears times is greater than or equal to its value

2. 'largest': means that we need the bigest interger that appears greater or equal to its value

3. There are no steps that could be implement simultaneously. All 4 steps shoule be implement step by step.

4. search([2,2,3,3,3]) = 3
   explanation: Both 2 and 3 appears greater than or equal to its value, but 3 is the largest number
   search([3,3,2,4,4,4]) = -1
   explanation: number 2 appears one time, number 3 appears two times,number 4 appears three times, none of them appears greater than or equal to its value, so the function return -1

5. We do not need extra code to handle the edge case. We could set the original return answer to -1 and then find the largest interger that meets the need.

## Prompt 3:
requirement:
<r>
plan
<p>

## Completion 3:
'''

def repair_plan_add_code_analysis(plan,requirement,model = 'gpt-35-turbo'):
    prompt = REPAIR_PROMPT_W_CORE_COMCEPT.replace('<r>',requirement).replace('<p>',plan)
    # print(prompt)
    message = [
        {"role": "user", "content": prompt}
    ]
    completions = []
    for _ in range(3):
        try:      
            requested_completions=1
            max_tokens=512
            temperature=0
            num_completions=1
            response = client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature,
            n=requested_completions
            )
            completions.extend([choice.message.content for choice in response.choices])
            if len(completions) >= num_completions:
                # print(completions[0])
                return completions[:num_completions]
        except Exception:
            time.sleep(20)
    return ''



MONITOR_SYSTEM_PROPMT = '''
You are a process monitor for the interaction process of a coding requirement analyst and a programmer. The task of coding requirement analyst is to write requirement coding plan for the programmer, and the task of programmer is to write python code based on the user's requirement and coding plan from analyst. 
'''
TASK_REPAIR_PLAN = '''
Now you receive a coding plan from the analyst and the original requirement from user, you task is to judge whether the plan need further inteperate based on the following perspective. If the plan need further inteperate, please provide some insight for the coder based on the following perspective.
1. Identift the core concept(key words, important concept) of the requirement, and explain the meaning of core concept.
2. Identify all the phrase showing quantity relationship (greater than, more than, two times, two multiply two, as much as) in the requirement, and explain the meaning of them in the requirement,then show how to implement them in code.
3. Identify all degree adverb (largest, greatest, best, shorest) in the requirement, and explain the meaning of them  in the requirement, then show how to implement them in code.
4. For the steps in plan, check if some steps should be implement simultaneously (in one code block or if-else statement), and explain the implementation
5. Based on the requirement and analysis, identify the edge case of the question, generate three edge case based on the format of edge cases in the requirement, and identify the correct output of edge case and explain it.
6. Based on the requirement and analysis, identify if extra code needed to handle the edge cases, or it could be solved in by considering original logic.

Noted that you should output 'YES' or 'NO' after your answer indicating whether the plan need further inteperate

- The format of your output should be:

# [core concept]
    <core concept>
    Explanation: ...

# [phrase showing quantity relationship]
    <phrase1>: <explanation> 
   ...
   
# [degree adverb] 
    <degree adverb1>: <explanation> 
   ... 

# [code logic]
(check if there are steps should be considered simultaneously)

# [edge case] 
    <edge case1> = <expected output>
    Explanation:
    ...

# [extra code for edge case]
    We need extra code to handle the edge cases.
        (code for handling the edge case)



# For example:
## Prompt 1:

[requirement]
def how_many_times(string: str, substring: str) -> int:
\'\'\' Find how many times a specific substring appears within the original string. Include overlapping instances.
>>> how_many_times('', 'a')
    0
    >>> how_many_times('aaa', 'a')
    3
    >>> how_many_times('aaaa', 'aa')
    3
    \'\'\'
[plan]
{
  "plan": {
    "subproblems": [
      "Identify the length of the original string",
      "Identify the length of the substring",
      "Iterate through the original string to find all occurrences of the substring",
      "Count the number of occurrences found"
    ],
    "steps": [
      "Get the input string and substring from the user",
      "Initialize a counter variable to keep track of the number of occurrences",
      "Iterate through the original string using a sliding window approach",
      "Check if the current substring matches the input substring",
      "If a match is found, increment the counter variable",
      "Return the final count of occurrences"
    ]
  }
}

## Answer 1:

# [core concept]
    'overlapping'
    In the requirement it means that we could count the overlapping apperance of substring in the original string

# [phrase showing quantity relationship]
    No phrase showing quantity relationship

# [degree adverb] 
    No degree adverb

# [code logic]
    The step 3-5 should be implement simultaneously
    "Iterate through the original string using a sliding window approach",
    "Check if the current substring matches the input substring",
    "If a match is found, increment the counter variable"
    This could be done by writing one for loop to iterate through the orginal string, extract every substring with the size of substring, check if it match the input substring and increment the counter variable if a match is found

# [edge case] 
    how_many_times('', 'a') = 0
    explanation: Since the original string is empty, the substring cannot appear, so the expected output is 0.
    how_many_times('abc', '') = 4
    explanation: '' appears four times in the orginal string. 'abc'.count('')=2

# [extra code for edge case]
    Extra code are needed to handle the edge case.
        if not string:
            return 0
        elif not substring:
            return len(string)+1
        (other code)


## Prompt 2:

[requirement]
def search(lst):	
\'\'\'You are given a non-empty list of positive integers. Return the largest integer that is more than zero and appears at least as many times as the integer itself. If no such a value exist, return -1.
        search([4, 1, 2, 2, 3, 1]) == 2
        search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3
        search([5, 5, 4, 4, 4]) == -1
    \'\'\'
[plan]
{
  "plan": {
    "subproblems": [
      "Identify the frequency of each integer in the list",
      "Find the largest integer that appears at least as many times as itself",
      "Handle the case where no such integer exists"
    ],
    "steps": [
      "Create a dictionary to store the frequency of each integer in the list",
      "Iterate through the list and update the frequency in the dictionary",
      "Iterate through the dictionary to find the largest integer that meets the condition",
      "Return the result or -1 if no such integer exists"
    ]
  }
}
}

## Answer 2:

# [core concept] 
    'positive': means that all interger in the list is > 0

    'at least as many times': means appears of a number >= its value

# [phrase showing quantity relationship]
    'more than': means that we need to find interger > 0
    'at least as many times': means that we need to find the interger whose appears times is greater than or equal to its value

# [degree adverb] 
    'largest': means that we need the bigest interger that appears greater or equal to its value

# [code logic]
    There are no steps that could be implement simultaneously. All 4 steps shoule be implement step by step.

# [edge case] 
    search([2,2,3,3,3]) = 3
    explanation: Both 2 and 3 appears greater than or equal to its value, but 3 is the largest number
    search([3,3,2,4,4,4]) = -1
    explanation: number 2 appears one time, number 3 appears two times,number 4 appears three times, none of them appears greater than or equal to its value, so the function return -1

# [extra code for edge case]
    We do not need extra code to handle the edge case. We could set the original return answer to -1 and then find the largest interger that meets the need. 

## Prompt 3:
[requirement]
<r>
[plan]
<p>

## Answer 3:

'''

TASK_JUDGE_CODE = '''
Now you receive a python code generated by the programmer, and the plan written by analyst as well as the original question. Your task is to judge whether the code follow the plan. If not, please explain the code's misunderstanding code to the plan. Your judgement should base on the following perspective. 
1. Does the code correctly understand the core concept of the plan?
2. Can the code handle all the edge cases provided in the plan?
Noted that you should output 'YES' or 'NO' 
[YES] indicates that the code contain misunderstanding of plan, need regenerate
[NO] indicates that the code does not contain misunderstanding of plan, do not need regenerate
If your answer is yes, please write suggestions for the programmer to better understand the plan

- The format of your output should be:

[YES] / [NO]

(if the answer is yes)
[suggestions]
1. ...


## Example
## Prompt 1:
[requirement]
def circular_shift(x, shift):
\'\'\'Circular shift the digits of the integer x, shift the digits right by shift
    and return the result as a string.
    If shift > number of digits, return digits reversed.
    Ensure that the result maintains any leading zeros from the original number.
>>> circular_shift(12, 1)
    "21"
    >>> circular_shift(12, 2)
    "12"
    \'\'\'
[plan]
{
  "plan": {
    "subproblems": [
      "Identify the number of digits in the input integer x",
      "Determine the actual number of shifts needed based on the input shift value",
      "Perform circular shifting of the digits to the right by the determined number of shifts",
      "Handle cases where the shift value is greater than the number of digits in the input integer"
    ],
    "high-level steps": [
      "Get the input integer x and shift value from the user",
      "Calculate the number of digits in the input integer x",
      "Determine the actual number of shifts needed based on the input shift value",
      "Perform circular shifting of the digits to the right by the determined number of shifts",
      "Handle cases where the shift value is greater than the number of digits in the input integer",
      "Return the result as a string"
    ]
  }
}
[code from programmer]
def circular_shift(x, shift):
    x_str = str(x)
    num_digits = len(x_str)
    actual_shift = shift % num_digits
    if actual_shift == 0:
        return x_str
    else:
        shifted_str = x_str[-actual_shift:] + x_str[:-actual_shift]
        return shifted_str.zfill(num_digits)

## Answer 1:
[YES] 
[suggestions]
1. the programmer should first consider the edge case when the shift value is greater than the number of digits, and then consider the other condtions.


## Prompt 2:
[requirement]
<r>
[plan]
<p>
[code from programmer]
<c>

## Answer 2:
'''




def monitor(plan,requirement,code='',model = 'gpt-35-turbo',task = 'repair_plan'):
    if task == 'repair_plan':
        prompt = TASK_REPAIR_PLAN.replace('<r>',requirement).replace('<p>',plan)
    if task == 'judge_code':
        prompt = TASK_JUDGE_CODE.replace('<r>',requirement+'').replace('<p>',plan).replace('<c>',code)
    # print(prompt)
    message = [
        {"role": "system", "content": MONITOR_SYSTEM_PROPMT},
        {"role": "user", "content": prompt}
    ]
    completions = []
    for _ in range(3):
        try:      
            requested_completions=1
            max_tokens=512
            temperature=0
            num_completions=1
            response = client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature,
            n=requested_completions
            )
            completions.extend([choice.message.content for choice in response.choices])
            if len(completions) >= num_completions:
                # print(completions[0])
                return completions[:num_completions]
        except Exception:
            time.sleep(20)
    return ''


# SELECT_PROMPT='''
# I will give you one original coding question, and several other questions mutated from the original question. I want you choose one of mutants as the best question and output ONE INTERGER indicating the index of your chosen mutant. The best question should preserve the meaning of original question, without adding or deleting any condition and constrait, and easy for programmers to understand.


# # original_question
# <original_question>
# # mutated question
# <mutated_question>

# Noted that your should output ONE INTERGER indicating the index(start from 0) of mutant!
# '''


# def select_best_prompt(mutant_list,model,original_promblem):
#     mutants = ''
#     index = '#index: <index>\n'
#     for idx,mutant in enumerate(mutant_list):
#         mutants+=index.replace('<index>',str(idx))
#         mutants+=mutant+'\n'
#     prompt = SELECT_PROMPT.replace('<original_question>',original_promblem).replace('<mutated_question>',mutants)
#     # print('in select_best_prompt')
#     # print(prompt)
#     message = [
#         {"role": "user", "content": prompt}
#     ]
#     completions = []
#     for _ in range(20):
#         try:      
#             requested_completions=1
#             max_tokens=512
#             temperature=0
#             num_completions=1
#             response = client.chat.completions.create(
#             model=model,
#             messages=message,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             n=requested_completions
#             )
#             completions.extend([choice.message.content for choice in response.choices])
#             if len(completions) >= num_completions:
#                 # print(completions[0])
#                 output= completions[0]
#                 number_list = [str(i) for i in range(len(mutant_list))]
#                 for sl in output.split(' '):
#                     if sl[0] in number_list:
#                         index = int(sl[0])
#                         return index
#                 # index = clean_output(completions[0])
#                 # return index
#         except Exception:
#             time.sleep(20)
#     raise RuntimeError('Failed to call GPT API')

mutate_prompt_nl_map = {'add_1_sentence_at_end':NL_ADD_1_SENTANCE_AT_END,'rephrase':NL_REPHRASE,'shorten':NL_SHORTEN, 'expand_one':NL_EXPAND_ONE_SENTENCE,'condense_one':NL_CONDENSE_ONE_SENTENCE,'expand_one2two':NL_EXPAND_ONE_SENTENCE_INTO_TWO,'condense_two2one':NL_CONDENSE_TWO_SENTENCE_INTO_ONE,'rephrase_one':NL_REPHRASE_ONE_SENTENCE}       
def test_mutate(intent):
    mutate_prompt = NL_CONDENSE_ONE_SENTENCE
    mutated_prompt = mutate('gpt-4o',intent,mutate_prompt)
    return mutated_prompt




# input = '''
# def has_close_elements(numbers: List[float], threshold: float) -> bool:
#     """ Check if in given list of numbers, are any two numbers closer to each other than
#     given threshold.
#     >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
#     False
#     >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
#     True
#     """
# '''

# input = '''
# "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n"
# '''
# # input = '''
# # "from typing import List\n\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n"
# # '''

# input ='''
# from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses, which you need to separate into separate strings and return the list of those. Separate groups are balanced (each open brace is properly closed) and not nested within each other. Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    """\n
# '''

# input ='''
# You have been tasked to write a function that receives \n    a hexadecimal number as a string and counts the number of hexadecimal \n    digits that are primes (prime number, or a prime, is a natural number \n    greater than 1 that is not a product of two smaller natural numbers).\n    Hexadecimal digits are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F.\n    Prime numbers are 2, 3, 5, 7, 11, 13, 17,...\n    So you have to determine a number of the following digits: 2, 3, 5, 7, \n    B (=decimal 11), D (=decimal 13).\n    Note: you may assume the input is always correct or empty string, \n    and symbols A,B,C,D,E,F are always uppercase.\n '''
# # ['from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    """ This function takes as input a string with several groups of nested parentheses, and your task is to split them into individual strings and return those as a list. Separate groups are balanced (each open brace is properly closed) and not nested within each other. Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    """']
# print(test_mutate(input))