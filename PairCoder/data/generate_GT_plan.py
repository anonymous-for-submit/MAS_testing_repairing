import json
import openai
from openai import OpenAI
from openai import AzureOpenAI
import time
import os
file  ='/data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl'
file2 = '/data/zlyuaj/muti-agent/fuzzing/output_mutated/original/code_round_0_with_score.jsonl'
PROMPT='''
There is a development team that includes a requirement analyst, a Python developer, and a tester. The team needs to develop programs that satisfy the requirement of the users. The different roles have different divisions of labor and need to cooperate with each others.\n\n The requirement from users and the canonical solution code is: \n{'requirement and solution':\n'<question_solution>'\n}\n\n.. I want you to act as a requirement analyst on our development team. Given a user requirement and its canonical solution code , your task is to analyze, decompose, and develop a high-level and concise plan to guide our developer in writing programs. The plan should include the following information:\n1. Decompose the requirement into several easy-to-solve subproblems that can be more easily implemented by the developer.\n2. Develop a high-level plan that outlines the major steps of the program.\nRemember, you only need to provide the concise plan in json.\n
'''

# lines_with_original_plan = []
# with open (file2,'r') as f:
#     lines_with_original_plan = [json.loads(line) for line in f]


with open('a.jsonl','w+') as f2:
    with open(file,'r') as f:
        lines = [json.loads(line) for line in f]
        for idx,task in enumerate(lines):
            print('handing task No. {}'.format(idx))
            GT_solution = task['canonical_solution']
            question = task['prompt']
            model='gpt-35-turbo'

            client = AzureOpenAI(
            azure_endpoint = "https://hkust.azure-api.net", 
            api_key="b234b6eb250e445d8151e8e5710dadde",  
            api_version="2024-02-01"
            )
            completions=[]
            prompt = PROMPT.replace('<question_solution>',question+GT_solution)
            # print(prompt)
            input_prompt = message = [
                {"role": "user", "content": prompt}
            ]
            num_completions=1
            for _ in range(num_completions):
                try:
                    # print('***'*30)
                    # print(prompt)
                    requested_completions = 1
                    # print(client.api_key)
                    # print(client.base_url)
                    # print(max_tokens,temperature,top_p,requested_completions)
                    response = client.chat.completions.create(
                        model=model,
                        messages=input_prompt,
                        max_tokens=512,
                        temperature=0,
                        top_p=0.95,
                        n=requested_completions
                        )
                    while not response:
                        response = client.chat.completions.create(
                            model=model,
                            messages=input_prompt,
                            max_tokens=512,
                            temperature=0,
                            top_p=0.95,
                            n=requested_completions
                            )

                    completions.extend([choice.message.content for choice in response.choices])
                except openai.RateLimitError as e:
                    time.sleep(1)
            task['canonical_plan']=completions
            lines_with_original_plan[idx]['canonical_plan']=completions
            f2.write(json.dumps(task) + '\n')
            # break


file3 = '/data/zlyuaj/muti-agent/fuzzing/output_mutated/original/a.jsonl'
 
with open (file3,'w+') as f:
    for task in lines_with_original_plan:
        f.write(json.dumps(task) + '\n')

# completions=[]
# with open(file,'r') as f:
#     lines = [json.loads(line) for line in f]
#     for idx,task in enumerate(lines):
#         if idx!=68:
#             continue
#         print('handing task No. {}'.format(idx))
#         GT_solution = task['canonical_solution']
#         question = task['prompt']
#         model='gpt-35-turbo'

#         client = AzureOpenAI(
#         azure_endpoint = "https://hkust.azure-api.net", 
#         api_key="b234b6eb250e445d8151e8e5710dadde",  
#         api_version="2024-02-01"
#         )
        
#         prompt = PROMPT.replace('<question_solution>',question+GT_solution)
#         # print(prompt)
#         input_prompt = message = [
#             {"role": "user", "content": prompt}
#         ]
#         num_completions=1
#         for _ in range(num_completions):
#             try:
#                 # print('***'*30)
#                 # print(prompt)
#                 requested_completions = 1
#                 # print(client.api_key)
#                 # print(client.base_url)
#                 # print(max_tokens,temperature,top_p,requested_completions)
#                 response = client.chat.completions.create(
#                 model=model,
#                 messages=input_prompt,
#                 max_tokens=512,
#                 temperature=0,
#                 top_p=0.95,
#                 n=requested_completions
#                 )
#                 completions.extend([choice.message.content for choice in response.choices])
#             except openai.RateLimitError as e:
#                 time.sleep(1)
#         task['canonical_plan']=completions
#         print(completions)
plans=[]
lines=[]
with open ('a.jsonl','r') as f:
    lines = [json.loads(line) for line in f]
    plans = [line['canonical_plan'] for line in lines]
with open (file2,'r') as f:
    lines = [json.loads(line) for line in f]
with open('/data/zlyuaj/muti-agent/fuzzing/output_mutated/original/a.jsonl','w+') as f:
    for i,task in enumerate(lines):
        task['canonical_plan']=plans[i]
        f.write(json.dumps(task) + '\n')

    
