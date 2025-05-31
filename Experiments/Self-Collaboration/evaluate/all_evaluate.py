import re
import json
import copy
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import build_test_method, find_method_name, code_split, prompt_split_humaneval, build_test_method_for_one_test
from execute.execution import evaluate_with_test_code, evaluate_with_test_code_T
from evaluation import pass_at_K, AvgPassRatio
from datasets import load_dataset, load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='humaneval')
parser.add_argument('--lang', type=str, default='python')
parser.add_argument('--input_path', type=str, default='humaneval_output_240415.jsonl')
parser.add_argument('--output_path', type=str, default='outputs/test_eval.jsonl')
args = parser.parse_args()


def evaluate_all(handled_solutions):
    print(len(handled_solutions))
    for solution in handled_solutions:
        solution["generation"] = solution['prompt'] + solution["completion"]  
        solution["prompt"] = ""
        solution["entry_point"] = find_method_name(solution["generation"]) if find_method_name(solution["generation"]) else "candidate"
        solution["completion"] = solution["generation"]

    print(INPUTPATH)
    data_dict = {}
    for key in dataset_key:
        for idx, task in enumerate(dataset[key]):
            data_dict[task['task_id']] = task

    # 根据读出的字段计算,返回的是原输入加上通过信息
    # 第一个是原test
    exec_result = evaluate_with_test_code(handled_solutions, timeout=10)
    # for i in exec_result:
    #     print('--'*30)
    #     print(i['task_id'])
    #     print(i['result'])
    #     print(i['passed'])
    print('pass@1:')
    ans1= pass_at_K(exec_result, k=[1])


    # 第二个是加入后的test（ET）
    if args.dataset == "humaneval":
        test_case_path= 'data/HumanEval_test_case_ET.jsonl'
        with open(test_case_path, 'r') as f:
            test_cases = [json.loads(line) for line in f]
            
        test_cases_dict = {}
        for case in test_cases:
            test = build_test_method(case['test_case_list'], "", case['entry_point'])
            test_cases_dict[case['task_id']] = test

    # 这一步实质上已经将
    for solution in handled_solutions:
        solution['test'] =test_cases_dict[solution['task_id']]

    exec_result_T = evaluate_with_test_code(handled_solutions, timeout=10)

    print('pass@1 - ET:')
    ans2=pass_at_K(exec_result_T, k=[1])
    
    return 


INPUTPATH = args.input_path


print(args.dataset)
if args.dataset == 'humaneval':
    dataset = load_dataset("openai_humaneval")
    dataset_key = ["test"]


with open(INPUTPATH, 'r') as f:
    except_list = []
    # 导入输出
    handled_solutions = [json.loads(line) for line in f if json.loads(line)["task_id"] not in except_list]
    print(len(handled_solutions))

# 分类字段 
# for solution in handled_solutions:
#     solution["generation"] = solution['prompt'] + solution["completion"]  
#     solution["prompt"] = ""
#     solution["entry_point"] = find_method_name(solution["generation"]) if find_method_name(solution["generation"]) else "candidate"
#     solution["completion"] = solution["generation"]

print(INPUTPATH)

print(len(handled_solutions))
for solution in handled_solutions:
    solution["generation"] = solution["completion"]  
    solution["prompt"] = ""
    solution["entry_point"] = find_method_name(solution["generation"]) if find_method_name(solution["generation"]) else "candidate"
    solution["completion"] = solution["generation"]

# 原有的test
# exec_result = evaluate_with_test_code(handled_solutions, timeout=10)
# print('pass@1:')
# ans1= pass_at_K(exec_result, k=[1])


# 第二个是加入后的test（ET）
if args.dataset == "humaneval":
    test_case_path= 'data/HumanEval_test_case_ET.jsonl'
    with open(test_case_path, 'r') as f:
        test_cases = [json.loads(line) for line in f]
        
    test_cases_dict = {}
    for case in test_cases:
        # 对每一个test case建立独立函数
        tests=[]
        for single_test in case['test_case_list']:
            test = build_test_method_for_one_test(single_test, "", case['entry_point'])
            tests.append(test)
        test_cases_dict[case['task_id']] = tests

# 这一步实质上已经将
for solution in handled_solutions:
    solution['test'] =test_cases_dict[solution['task_id']]

exec_result_T = evaluate_with_test_code(handled_solutions, timeout=10)
# print(exec_result_T)
scores=[sample['score'] for sample in exec_result_T]
print('pass@1 - ET:')
ans=pass_at_K(exec_result_T, k=[1])

print( ans, scores)


OUTPUT_PATH = args.output_path + args.input_path.split('/')[-1]
with open(OUTPUT_PATH, 'w+') as f:
    results = {
        'pass_1':ans,
        'scores':scores
    }
    
    
    f.write(json.dumps(results) + '\n')
    f.flush()
# data_dict = {}
# for key in dataset_key:
#     for idx, task in enumerate(dataset[key]):
#         data_dict[task['task_id']] = task

# # 根据读出的字段计算,返回的是原输入加上通过信息
# # 第一个是原test
# exec_result = evaluate_with_test_code(handled_solutions, timeout=10)
# # for i in exec_result:
# #     print('--'*30)
# #     print(i['task_id'])
# #     print(i['result'])
# #     print(i['passed'])
# print('pass@1:')
# scores=[sample['score'] for sample in exec_result]
# print(scores)
# pass_at_K(exec_result, k=[1])


# # 第二个是加入后的test（ET）
# if args.dataset == "humaneval":
#     test_case_path= 'data/HumanEval_test_case_ET.jsonl'
#     with open(test_case_path, 'r') as f:
#         test_cases = [json.loads(line) for line in f]
        
#     test_cases_dict = {}
#     for case in test_cases:
#         test = build_test_method(case['test_case_list'], "", case['entry_point'])
#         test_cases_dict[case['task_id']] = test

# # 这一步实质上已经将
# for solution in handled_solutions:
#     solution['test'] =test_cases_dict[solution['task_id']]

# exec_result_T = evaluate_with_test_code(handled_solutions, timeout=10)

# print('pass@1 - ET:')
# scores=[sample['score'] for sample in exec_result_T]
# print(scores)
# pass_at_K(exec_result_T, k=[1])
# for i in exec_result:
#     print('--'*30)
#     print(i['task_id'])
#     print(i['result'])
#     print(i['passed'])

# no_pass=[]
# no_pass_T=[]
# for i in range(len(exec_result)):
#     x=exec_result[i]
#     if x['passed']!=True:
#         no_pass.append(i)
# for i in range(len(exec_result_T)):
#     x=exec_result_T[i]
#     if x['passed']!=True:
#         no_pass_T.append(i)
# print(len(no_pass))
# print(len(no_pass_T))
# a=set()
# for i in no_pass+no_pass_T:
#     a.add(i)
# print(a)
# print(len(a))



# gpt3.5 ca output.json
# SystemLog: [2024-10-30 10:29:45][evaluation][INFO] - {'pass@1': 0.7012}
# SystemLog: [2024-10-30 10:29:57][evaluation][INFO] - {'pass@1': 0.6402}
'''

59
59
{129, 2, 130, 131, 132, 135, 137, 10, 11, 12, 138, 140, 142, 145, 151, 25, 26, 154, 159, 32, 163, 38, 39, 41, 44, 50, 53, 54, 55, 64, 65, 67, 75, 76, 77, 80, 81, 83, 84, 86, 87, 90, 91, 93, 94, 95, 96, 97, 99, 100, 101, 108, 111, 115, 118, 119, 120, 125, 126}
59

'''