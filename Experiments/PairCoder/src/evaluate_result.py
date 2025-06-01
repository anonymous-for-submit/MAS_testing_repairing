import re
import json
import copy
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append('/data/zlyuaj/muti-agent/PairCoder/')
sys.path.append('/data/zlyuaj/muti-agent/PairCoder/src/evaluate/')
sys.path.append('/data/zlyuaj/muti-agent/PairCoder/src/')
from execution import evaluate_with_test_code_one_sample
from datasets import load_dataset, load_from_disk
from concurrent.futures import as_completed, ProcessPoolExecutor
import multiprocessing
import subprocess
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
def find_method_name(code, lang="python"):
    try:
        parsed = ast.parse(code)
        function_defs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        if function_defs:
            if len(function_defs) == 1:
                method_name = function_defs[0].name
            else:
                method_name = function_defs[-1].name if function_defs[-1].name != "main" else function_defs[-2].name
        else:
            method_name = None
    except:
        method_name = None

    return method_name
def build_test_method_for_one_test(test, test_imports, method_name):
    if test_imports:
        test_imports = "\n".join(test_imports)
        test_method = test_imports + "\n"
    else:
        test_method = ""
    test_method = "def check(" + method_name + "):\n"
    if len(test) == 0:
        return test_method + "\treturn True" + "\n"
    test_method += '\t' + test + "\n"
    return test_method.strip("\n")


CODE_HEAD= 'from typing import *\n'

def evaluate_one(solution,num_generate):
    solution["completions"] = [CODE_HEAD+completion for completion in solution["completions"]]
    for i in range(num_generate):
        if "entry_point" not in solution.keys() or not solution["entry_point"]:
            solution["entry_point"] = find_method_name(solution["completions"][i]) 
    
    if "entry_point" not in solution.keys() or not solution["entry_point"]:
        print('#'*20)
        print(solution["prompt"])
        print(solution["completions"][0])
        print('#'*20)
        print(solution["entry_point"])
        return -1,0,False, False

    
    
    if 'test_case_list' not in solution.keys():
        task_num = int(solution['task_id'].split('/')[-1])
        test_case_path= '/data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl'
        with open(test_case_path, 'r') as f:
            test_cases = [json.loads(line) for line in f]
            task_dict = {task['task_id']:task for task in test_cases}
            solution['test_case_list'] = task_dict[solution['task_id']]['test_case_list']



    if 'test' not in solution.keys() or type(solution['test'])!=list:
        tests=[]
        for single_test in solution['test_case_list']:
            test = build_test_method_for_one_test(single_test, "", solution['entry_point'])
            tests.append(test)

        solution['test'] =tests


    solution['scores']=[]
    solution['pass_results']=[]
    solution['pass_test_cases_num']=[]

    for i in range(num_generate):
        if 'completions_after_repair' in solution.keys():
            solution['completion']=solution['completions_after_repair'][i]
        else:
            solution['completion']=solution['completions'][i]
        # print(solution['completion'])
        # print('number {}'.format(i))
        solution = evaluate_with_test_code_one_sample(solution, timeout=10)
        # print(solution['completion'])
        # print(solution['scores'])
        # print(solution)

    # print(solution['pass_results'])
    # print(solution['scores'])
    # print(solution['pass_test_cases_num'])

    passAt1, passAt10 = False, False
    passAt1 = solution['pass_results'][0]
    if True in solution['pass_results']:
        solution['passed']=True
        passAt10 = True
    else:
        solution['passed']=False
        passAt10 = False
    # max_score = max(solution['scores'])
    # 以最高分作为代表分
    max_score,index = max((a,i) for (i,a) in enumerate(solution['scores']))
    score=round(sum(solution['scores'])/len(solution['scores']),4)
    # passes=round(sum(solution['pass_test_cases_num'])/num_generate,4)
    passes=solution['pass_results'].count(True)
    # 找到最高分的代码
    # solution['completion']=solution['completions'][index]
    # solution['session_history']=solution['session_historys'][index]


    # OUTPUT_PATH = './output_fuzzing_one_per_time/result.jsonl'
    # print(OUTPUT_PATH) 
    # with open(OUTPUT_PATH, 'a') as f:
    #     results = {
    #         'score':score,
    #         'passes':passes,
    #         'max_scores':max_score
    #     }
        
        
    #     f.write(json.dumps(results) + '\n')
    #     f.flush()
    
    # avg score 为所有题目的得分均值，avg scores为每个题目的多次生成后的得分均值， avg passes为每个题目平均通过的样例数量
    return score, passes, passAt1, passAt10

def evaluate_one_MBPP(solution,num_generate):
    solution["completions"] = [CODE_HEAD+completion for completion in solution["completions"]]
    # task_num = int(solution['task_id'].split('/')[-1])
    # test_case_path= '/data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl'
    # with open(test_case_path, 'r') as f:
    #     test_cases = [json.loads(line) for line in f]
    #     solution['test_case_list'] = test_cases[task_num]['test_case_list']




    tests=[]
    for single_test in solution['test_list']:
        test = build_test_method_for_one_test(single_test, "", solution['entry_point'])
        tests.append(test)

    solution['test'] =tests
    if 'prompt' not in solution.keys():
        solution['prompt']=solution['text']

    solution['scores']=[]
    solution['pass_results']=[]
    solution['pass_test_cases_num']=[]

    for i in range(num_generate):
        # solution['completion']=solution['completions'][i]
        if 'completions_after_repair' in solution.keys():
            solution['completion']=solution['completions_after_repair'][i]
        else:
            solution['completion']=solution['completions'][i]
        # print(solution['completion'])
        # print('number {}'.format(i))
        solution = evaluate_with_test_code_one_sample(solution, timeout=10)
        # print(solution['completion'])
        # print(solution['scores'])
        # print(solution)

    # print(solution['pass_results'])
    # print(solution['scores'])
    # print(solution['pass_test_cases_num'])

    passAt1, passAt10 = False, False
    passAt1 = solution['pass_results'][0]
    if True in solution['pass_results']:
        solution['passed']=True
        passAt10 = True
    else:
        solution['passed']=False
        passAt10 = False
    # max_score = max(solution['scores'])
    # 以最高分作为代表分
    max_score,index = max((a,i) for (i,a) in enumerate(solution['scores']))
    score=round(sum(solution['scores'])/len(solution['scores']),4)
    # passes=round(sum(solution['pass_test_cases_num'])/num_generate,4)
    passes=solution['pass_results'].count(True)
    solution['test'] =''
    # 找到最高分的代码
    # solution['completion']=solution['completions'][index]
    # solution['session_history']=solution['session_historys'][index]


    # OUTPUT_PATH = './output_fuzzing_one_per_time/result.jsonl'
    # print(OUTPUT_PATH) 
    # with open(OUTPUT_PATH, 'a') as f:
    #     results = {
    #         'score':score,
    #         'passes':passes,
    #         'max_scores':max_score
    #     }
        
        
    #     f.write(json.dumps(results) + '\n')
    #     f.flush()
    
    # avg score 为所有题目的得分均值，avg scores为每个题目的多次生成后的得分均值， avg passes为每个题目平均通过的样例数量
    return score, passes, passAt1, passAt10

