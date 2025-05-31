import os
import copy
import json
import argparse
import tqdm
import random
# from session import Session
from datasets import load_dataset, load_from_disk
from core.backend import call_deepseek_coder
from utils import prompt_split_humaneval, find_method_name, code_split, build_test_method
import copy
import time
from main_fuzz_passAt10 import PromptNode
# from main_fuzz_passAt10 import prompt

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

# CHANGE_IDENTIFIER_FUNCNAME = '''
# I will give you a coding question prompt, with several test cases. You are required to change the identifier of the given code into random strings, while remaining other sentences unchanged. Do not change the function name! Also, you should maintain the overall meaning of the template and SHOULD NOT delete the test cases. Do not make any change to the meaning of the question. You should not change the input format and output format. Do not make any other explanation nor have beginning or ending indicator in your answer. 
# Here is the question:
# '''


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
        azure_endpoint = "your end point", 
        api_key="your api key",  
        api_version="your api version"
    )

def mutate(model,prompt,mutate_prompt):
    prompt=mutate_prompt+prompt
    message = [
        {"role": "user", "content": prompt}
    ]
    if 'deepseek' in model:
        return call_deepseek_coder(message,model)
    completions = []
    for _ in range(5):
        try:      
            requested_completions=1
            max_tokens=1024
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
        except Exception:
            time.sleep(10)
    return ['']
    # raise RuntimeError('Failed to call GPT API')


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

    if 'MBPP' not in args.dataset:
        while 'def ' not in mutated_prompt:
            print('改变了prompt的结构!!!')
            mutated_prompt = mutate(model,intent,prompt4mutation)[0]

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

    new_prompt = task['func']+'\t\n\'\'\''+mutated_nl+'\n'+task['examples'] +'\'\'\''
    
    return new_prompt

def get_more_prompt_test(prompt,args,mutate_method='random',model='gpt-4o'):
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


# REPAIR_PROMPT_W_CORE_COMCEPT = '''
# Here is a coding requirement from the user and plan from analyst
# requirement:
# <r>
# plan
# <p>
# I want you to read the plan and requirement, and provide some insight for the coder based on the follow perspective.
# Noted that you should answer precise and correct.
# 1. Identift the core concept(key words, important concept) of the requirement, and explain the meaning of core concept.
# 2. Identify all the phrase showing quantity relationship (greater than, more than, two times, two multiply two, as much as) in the requirement, and explain the meaning of them in the requirement,then show how to implement them in code.
# 3. Identify all degree adverb (largest, greatest, best, shorest) in the requirement, and explain the meaning of them  in the requirement, then show how to implement them in code.
# 4. For the steps in plan, check if some steps should be implement simultaneously (in one code block or if-else statement), and explain the implementation
# 5. Based on the requirement and analysis, identify the edge case of the question, generate three edge case based on the format of edge cases in the requirement, and identify the correct output of edge case and explain it.
# 6. Based on the requirement and analysis, Do we need extra code to handle the edge cases, or it could be solved in original code? If extra code is needed, please write sample code to handle the edge case. PLEASE ONLY WRITE THE CODE HANDLING THE EDGE CASE!

# - The format of test cases should be:
# 1. core concept: <core concept>
#    Explanation: ...
# 1. <phrase>: <explanation> 
#    ...
# 2. <degree adverb>: <explanation> 
#    ... 
# 4. (check if there are steps should be considered simultaneously)
# 5. <edge case> = <expected output>
#    Explanation:
#    ...
# 6. We need extra code to handle the edge cases.
#     (code for handling the edge case)

# # For example:
# ## Prompt 1:
# requirement:
# def how_many_times(string: str, substring: str) -> int:
# \'\'\' Find how many times a specific substring appears within the original string. Include overlapping instances.
# >>> how_many_times('', 'a')
#     0
#     >>> how_many_times('aaa', 'a')
#     3
#     >>> how_many_times('aaaa', 'aa')
#     3
#     \'\'\'
# plan
# {
#   "plan": {
#     "subproblems": [
#       "Identify the length of the original string",
#       "Identify the length of the substring",
#       "Iterate through the original string to find all occurrences of the substring",
#       "Count the number of occurrences found"
#     ],
#     "steps": [
#       "Get the input string and substring from the user",
#       "Initialize a counter variable to keep track of the number of occurrences",
#       "Iterate through the original string using a sliding window approach",
#       "Check if the current substring matches the input substring",
#       "If a match is found, increment the counter variable",
#       "Return the final count of occurrences"
#     ]
#   }
# }

# ## Completion 1:
# 1. core concept: overlapping
#    In the requirement it means that we could count the overlapping apperance of substring in the original string

# 2. No phrase showing quantity relationship

# 3. No degree adverb

# 4. The step 3-5 should be implement simultaneously
#     "Iterate through the original string using a sliding window approach",
#     "Check if the current substring matches the input substring",
#     "If a match is found, increment the counter variable"
#     This could be done by writing one for loop to iterate through the orginal string, extract every substring with the size of substring, check if it match the input substring and increment the counter variable if a match is found

# 5. how_many_times('', 'a') = 0
#    explanation: Since the original string is empty, the substring cannot appear, so the expected output is 0.
#    how_many_times('abc', '') = 4
#    explanation: '' appears four times in the orginal string. 'abc'.count('')=4

# 6. Extra code are needed to handle the edge case.
#     if not string:
#         return 0
#     elif not substring:
#         return len(string)+1
#     (other code)


# ## Prompt 2:

# requirement:
# def search(lst):	
# \'\'\'You are given a non-empty list of positive integers. Return the largest integer that is more than zero and appears at least as many times as the integer itself. If no such a value exist, return -1.
#         search([4, 1, 2, 2, 3, 1]) == 2
#         search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3
#         search([5, 5, 4, 4, 4]) == -1
#     \'\'\'
# plan:
# {
#   "plan": {
#     "subproblems": [
#       "Identify the frequency of each integer in the list",
#       "Find the largest integer that appears at least as many times as itself",
#       "Handle the case where no such integer exists"
#     ],
#     "steps": [
#       "Create a dictionary to store the frequency of each integer in the list",
#       "Iterate through the list and update the frequency in the dictionary",
#       "Iterate through the dictionary to find the largest integer that meets the condition",
#       "Return the result or -1 if no such integer exists"
#     ]
#   }
# }
# }

# ## Completion 2:
# 1. core concept: positive, at least as many times
#    positive means that all interger in the list is > 0
#    at least as many times means appears of a number >= its value

# 2. 'more than': means that we need to find interger > 0
#    'at least as many times': means that we need to find the interger whose appears times is greater than or equal to its value

# 3. 'largest': means that we need the bigest interger that appears greater or equal to its value

# 4. There are no steps that could be implement simultaneously. All 4 steps shoule be implement step by step.

# 5. search([2,2,3,3,3]) = 3
#    explanation: Both 2 and 3 appears greater than or equal to its value, but 3 is the largest number
#    search([3,3,2,4,4,4]) = -1
#    explanation: number 2 appears one time, number 3 appears two times,number 4 appears three times, none of them appears greater than or equal to its value, so the function return -1

# 6. We do not need extra code to handle the edge case. We could set the original return answer to -1 and then find the largest interger that meets the need. 

# ## Prompt 3:
# requirement:
# <r>
# plan
# <p>

# ## Answer 3:
# '''
# REPAIR_PROMPT= '''
# Here is a coding requirement from the user and plan from analyst
# requirement:
# <r>
# plan
# <p>
# I want you to read the plan and requirement, and provide some insight for the coder based on the follow perspective.
# Noted that you should answer precise and correct.

# 1. Identify all the phrase showing quantity relationship (greater than, more than, two times, two multiply two, as much as) in the requirement, and explain the meaning of them in the requirement,then show how to implement them in code.
# 2. Identify all degree adverb (largest, greatest, best, shorest) in the requirement, and explain the meaning of them in the requirement, then show how to implement them in code.
# 3. For the steps in plan, check if some steps should be implement simultaneously (in one code block or if-else statement), and explain the implementation
# 4. Based on the requirement and analysis, identify the edge case of the question, generate three edge case based on the format of edge cases in the requirement, and identify the correct output of edge case and explain it.
# 5. Based on the requirement and analysis, Do we need extra code to handle the edge cases, or it could be solved in original code? If extra code is needed, please write sample code to handle the edge case. PLEASE ONLY WRITE THE CODE HANDLING THE EDGE CASE!

# - The format of test cases should be:
# 1. <phrase>: <explanation> 
#    ...
# 2. <degree adverb>: <explanation> 
#    ... 
# 3. (check if there are steps should be considered simultaneously)
# 4. <edge case> = <expected output>
#    Explanation:
#    ...
# 5. We need extra code to handle the edge cases.
#     (code for handling the edge case)

# # For example:
# ## Prompt 1:
# requirement:
# def how_many_times(string: str, substring: str) -> int:
# \'\'\' Find how many times a specific substring appears within the original string. Include overlapping instances.
# >>> how_many_times('', 'a')
#     0
#     >>> how_many_times('aaa', 'a')
#     3
#     >>> how_many_times('aaaa', 'aa')
#     3
#     \'\'\'
# plan
# {
#   "plan": {
#     "subproblems": [
#       "Identify the length of the original string",
#       "Identify the length of the substring",
#       "Iterate through the original string to find all occurrences of the substring",
#       "Count the number of occurrences found"
#     ],
#     "steps": [
#       "Get the input string and substring from the user",
#       "Initialize a counter variable to keep track of the number of occurrences",
#       "Iterate through the original string using a sliding window approach",
#       "Check if the current substring matches the input substring",
#       "If a match is found, increment the counter variable",
#       "Return the final count of occurrences"
#     ]
#   }
# }

# ## Completion 1:
# 1. No phrase showing quantity relationship

# 2. No degree adverb

# 3. The step 3-5 should be implement simultaneously
#     "Iterate through the original string using a sliding window approach",
#     "Check if the current substring matches the input substring",
#     "If a match is found, increment the counter variable"
#     This could be done by writing one for loop to iterate through the orginal string, extract every substring with the size of substring, check if it match the input substring and increment the counter variable if a match is found

# 4. how_many_times('', 'a') = 0
#    explanation: Since the original string is empty, the substring cannot appear, so the expected output is 0.
#    how_many_times('abc', '') = 4
#    explanation: '' appears four times in the orginal string. 'abc'.count('')=2

# 5. Extra code are needed to handle the edge case.
#     if not string:
#         return 0
#     elif not substring:
#         return len(string)+1
#     (other code)


# ## Prompt 2:

# requirement:
# def search(lst):	
# \'\'\'You are given a non-empty list of positive integers. Return the largest integer that is more than zero and appears at least as many times as the integer itself. If no such a value exist, return -1.
#         search([4, 1, 2, 2, 3, 1]) == 2
#         search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3
#         search([5, 5, 4, 4, 4]) == -1
#     \'\'\'
# plan:
# {
#   "plan": {
#     "subproblems": [
#       "Identify the frequency of each integer in the list",
#       "Find the largest integer that appears at least as many times as itself",
#       "Handle the case where no such integer exists"
#     ],
#     "steps": [
#       "Create a dictionary to store the frequency of each integer in the list",
#       "Iterate through the list and update the frequency in the dictionary",
#       "Iterate through the dictionary to find the largest integer that meets the condition",
#       "Return the result or -1 if no such integer exists"
#     ]
#   }
# }
# }

# ## Completion 2:
# 1. 'more than': means that we need to find interger > 0
#    'at least as many times': means that we need to find the interger whose appears times is greater than or equal to its value

# 2. 'largest': means that we need the bigest interger that appears greater or equal to its value

# 3. There are no steps that could be implement simultaneously. All 4 steps shoule be implement step by step.

# 4. search([2,2,3,3,3]) = 3
#    explanation: Both 2 and 3 appears greater than or equal to its value, but 3 is the largest number
#    search([3,3,2,4,4,4]) = -1
#    explanation: number 2 appears one time, number 3 appears two times,number 4 appears three times, none of them appears greater than or equal to its value, so the function return -1

# 5. We do not need extra code to handle the edge case. We could set the original return answer to -1 and then find the largest interger that meets the need.

# ## Prompt 3:
# requirement:
# <r>
# plan
# <p>

# ## Completion 3:
# '''

# def repair_plan_add_code_analysis(plan,requirement,model = 'gpt-35-turbo'):
#     prompt = REPAIR_PROMPT_W_CORE_COMCEPT.replace('<r>',requirement).replace('<p>',plan)
#     # print(prompt)
#     message = [
#         {"role": "user", "content": prompt}
#     ]
#     completions = []
#     for _ in range(3):
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
#                 return completions[:num_completions]
#         except Exception:
#             time.sleep(20)
#     return ''



