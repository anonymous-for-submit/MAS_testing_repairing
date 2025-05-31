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


import os
import copy
import json
import argparse
import tqdm
import random
# from session import Session
import copy
import time
import litellm
import openai
from openai import OpenAI
from openai import AzureOpenAI
from openai import AsyncOpenAI
import time
import os
# from fastchat.model import load_model, get_conversation_template
# from vllm import LLM as vllm
# from vllm import SamplingParams


async def call_deepseek_coder(prompt, model='deepseek-coder', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, echo=False, majority_at=None):
    if model == 'deepseek-coder':
        model = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    openai_api_key = "EMPTY"
    openai_api_base = "http://127.0.0.1:8000/v1/"

    aclient = AsyncOpenAI(
            base_url="http://127.0.0.1:8000/v1/", 
            api_key="EMPTY"  
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
            response = await aclient.chat.completions.create(
            model=model,
            messages=prompt,
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

async def monitor(plan,requirement,code='',model = 'gpt-35-turbo',task = 'repair_plan'):
    # print('in monitor!')
    # print(f'model = {model}')
    if task == 'repair_plan':
        prompt = TASK_REPAIR_PLAN.replace('<r>',requirement).replace('<p>',plan)
    if task == 'judge_code':
        prompt = TASK_JUDGE_CODE.replace('<r>',requirement+'').replace('<p>',plan).replace('<c>',code)
    # print(prompt)
    message = [
        {"role": "system", "content": MONITOR_SYSTEM_PROPMT},
        {"role": "user", "content": prompt}
    ]

    
    if 'deepseek' in model:
        # print('call deepseek model!')
        return await call_deepseek_coder(message,model)
    

    # print(prompt)
    completions = []
    for _ in range(3):
        try:      
            requested_completions=1
            max_tokens=512
            temperature=0
            num_completions=1
            response = await litellm.acompletion(
                        model = f"azure/{model}",
                        api_base = "https://hkust.azure-api.net",
                        api_version = "2024-02-01", 
                        api_key = "f9f10057a7e749898daeabdf5f6b84be",
                        messages=message,
                        temperature=temperature,
                        frequency_penalty=0.1,
                        n = num_completions, 
                        top_p=1,
                        force_timeout=90,
                    )
            completions.extend([choice.message.content for choice in response.choices])
            if len(completions) >= num_completions:
                # print(completions[0])
                return completions[:num_completions]
        except Exception:
            time.sleep(20)
    return ''

import asyncio
print(asyncio.run(monitor('','',task = 'judge_code')))

async def monitor_plan(plan,requirement,model):
        max_try=3
        for i in range(max_try):
            try:
                # print(f'in session.monitor plan, model = {self.model}')
                res = await monitor(plan,requirement,model=model,task= 'repair_plan')
                if type(res) == list and len(res)>0:
                    more_plan=res[0]
                    break
                else:
                    more_plan=''
            except Exception as e:
                print(e)
                more_plan=''
        if not more_plan:
            print('fail to generate interperated plan!')
            more_plan = ''
        INTEPERATE_PROMPT = '\nPlease read and understand the following inteperation before coding\n'
        # split_plan = more_plan.split('\n')
        # cleaned_plan = ''
        # for plan_line in split_plan:
        #     if '2.' in plan_line and ('No' in plan_line or 'no' in plan_line):
        #         continue
        #     if len(plan_line)>3 and 

        return plan + INTEPERATE_PROMPT +  more_plan

async def monitor_code(code,plan,requirement,model):
    try:
        # print(f'in session.monitor_code, model = {self.model}')
        res = await monitor(plan,requirement,code,model,task= 'judge_code')
        if type(res) == list and len(res)>0:
            code_analysis=res[0]
        else:
            code_analysis=''
    except Exception as e:
        print(e)
        code_analysis=''
    need_regenerate = False
    if type(code_analysis)==str and '[YES]' in code_analysis:
        need_regenerate = True
    return code_analysis ,need_regenerate