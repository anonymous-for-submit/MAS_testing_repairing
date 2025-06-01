import re
import json
import copy
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# from metagpt._utils import build_test_method, find_method_name, code_split, prompt_split_humaneval, build_test_method_for_one_test
from evaluate.execute.execution import evaluate_with_test_code, evaluate_with_test_code_T,evaluate_with_test_code_one_sample
from evaluate.evaluation import pass_at_K, AvgPassRatio
# from datasets import load_dataset, load_from_disk
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
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


def evaluate_all(handled_solutions,dataset,output_path,num_generate,cur_round):
    print(len(handled_solutions))
    for solution in handled_solutions:
        solution["prompt"] = ""
        solution["entry_point"] = find_method_name(solution["completion"]) if find_method_name(solution["completion"]) else "candidate"

    # 原有的test
    # exec_result = evaluate_with_test_code(handled_solutions, timeout=10)
    # print('pass@1:')
    # ans1= pass_at_K(exec_result, k=[1])


    # 第二个是加入后的test（ET）
    if dataset == "humaneval":
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
        solution['scores']=[]
        solution['pass_results']=[]
        solution['pass_test_cases_num']=[]

    for i in range(num_generate):
        for solution in handled_solutions:
            solution['completion']=solution['completions'][i]
        handled_solutions = evaluate_with_test_code(handled_solutions, timeout=10)


    max_scores=[]
    avg_scores=[]
    avg_passes=[]
    for solution in handled_solutions:
        # print('&&&'*10)
        # print(solution['pass_results'])
        # print(solution['scores'])
        if True in solution['pass_results']:
            solution['passed']=True
        else:
            solution['passed']=False
        max_score,index = max((a,i) for (i,a) in enumerate(solution['scores']))
        # max_score = max(solution['scores'])
        # 以最高分作为代表分
        max_scores.append(max_score)
        avg_scores.append(round(sum(solution['scores'])/len(solution['scores']),4))
        avg_passes.append(round(sum(solution['pass_test_cases_num'])/num_generate,4))
        # 找到最高分的代码
        solution['completion']=solution['completions'][index]
        solution['session_history']=solution['session_historys'][index]

    avg_score = round(sum(avg_scores)/len(avg_scores),4)

    print('pass@1 - ET:')
    ans=pass_at_K(handled_solutions, k=[1])
    # print(output_path)
    # print(output_path.split('/'))
    OUTPUT_PATH = './output_result/' + str(output_path.split('/')[-2]) + '/'+ str(cur_round) + '.jsonl'
    # print(OUTPUT_PATH) 
    with open(OUTPUT_PATH, 'w+') as f:
        results = {
            'pass_10':ans,
            'scores':avg_scores,
            'avg_score':avg_score,
            'avg_passes':avg_passes,
            'max_scores':max_scores
        }
        
        
        f.write(json.dumps(results) + '\n')
        f.flush()
    
    # avg score 为所有题目的得分均值，avg scores为每个题目的多次生成后的得分均值， avg passes为每个题目平均通过的样例数量
    return ans, avg_score,avg_scores, avg_passes,sum(avg_passes)



def evaluate_one(solution,num_generate):
    # solution["entry_point"] = find_method_name(solution["completions"][0]) 
    
    task_num = int(solution['task_id'].split('/')[-1])
    # test_case_path= '/data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl'
    # with open(test_case_path, 'r') as f:
    #     test_cases = [json.loads(line) for line in f]
    #     solution['test_case_list'] = test_cases[task_num]['test_case_list']



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
        if 'text' in solution.keys():
            solution['prompt']=solution['text']
        else:
            raise NotImplementedError

    solution['scores']=[]
    solution['pass_results']=[]
    solution['pass_test_cases_num']=[]

    for i in range(num_generate):
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
    solution['completion']=solution['completions'][index]
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

sys.path.append('/home/zlyuaj/muti-agent/PairCoder/')
sys.path.append('/home/zlyuaj/muti-agent/PairCoder/src/')
sys.path.append('/home/zlyuaj/muti-agent/PairCoder/src/code_contests/eval')
from concurrent.futures import as_completed, ProcessPoolExecutor
import multiprocessing
from code_test_runners import get_pass_result
def evaluate_one_codecontest(solution,num_generate):
    # solution["completions"] = [CODE_HEAD+completion for completion in solution["completions"]]
    
    
    solution['scores']=[]
    solution['pass_results']=[]
    solution['pass_test_cases_num']=[]
    futures = []
    # with ProcessPoolExecutor() as executor:
    #     for i in range(num_generate):
    #         future = executor.submit(run_code_with_test, solution['completions'][i],solution['test_list'],i)
    #         futures.append(future)
    # # logger.info(f'{len(futures)} execution requests are submitted')  
    # for idx, future in enumerate(as_completed(futures)):
    #     ans = future.result()
    #     solution['pass_results'].append(ans)
    inputs = [test['input'] for test in solution['test_list']]
    outputs = [test['output'][0] for test in solution['test_list']]



    

    # pass_result,pass_num = get_pass_result(solution['completions'], inputs, outputs)
    # print(solution['completions'][0])
    # print(inputs[0])
    # print(outputs[0])

    pass_num=[]
    pass_result=[]
    # for i in solution['completions']:
    #     print('-'*30)
    #     print(i)
    futures = []
    with ProcessPoolExecutor() as executor:
        for completion in solution['completions']:
                futures.append(executor.submit(get_pass_result,completion, inputs, outputs)) 
    for _, future in enumerate(as_completed(futures)):
        cur_ans  = future.result()
        pass_result.append(cur_ans)

            # for j in range(len(inputs)):
            #     cur_ans,cur_pass_num = get_pass_result([completion], inputs, outputs)
                # if not cur_ans[0]:
                #     pass_result.append(False)
                #     break
                # if j==len(inputs)-1:
                #     pass_result.append(True)

    print(pass_result)
    # print(pass_num)
    solution['pass_results']=pass_result
    passAt1, passAt10 = 0, False
    
    if True in solution['pass_results']:
        solution['passed']=True
        passAt10 = True
    else:
        solution['passed']=False
        passAt10 = False

    passes=solution['pass_results'].count(True)
    passAt1 = passes/num_generate
    return 0, passes, passAt1, passAt10


def evaluate_original(solution,num_generate):
    solution["entry_point"] = find_method_name(solution["completion"]) if find_method_name(solution["completion"]) else "candidate"
    
    
    # task_num = int(solution['task_id'].split('/')[-1])
    # test_case_path= '/data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl'
    # with open(test_case_path, 'r') as f:
    #     test_cases = [json.loads(line) for line in f]
    #     solution['test_case_list'] = test_cases[task_num]['test_case_list']




    tests=[]
    for single_test in solution['test']:
        test = build_test_method_for_one_test(single_test, "", solution['entry_point'])
        tests.append(test)

    solution['test'] =tests


    solution['scores']=[]
    solution['pass_results']=[]
    solution['pass_test_cases_num']=[]

    for i in range(num_generate):
        solution['completion']=solution['completions'][i]
        # print(solution['completion'])
        # print('number {}'.format(i))
        solution = evaluate_with_test_code_one_sample(solution, timeout=10)

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
    solution['completion']=solution['completions'][index]
    solution['session_history']=solution['session_historys'][index]

    
    # avg score 为所有题目的得分均值，avg scores为每个题目的多次生成后的得分均值， avg passes为每个题目平均通过的样例数量
    return score, passes, passAt1, passAt10,solution['pass_results']

# original_data_path= '/data/zlyuaj/muti-agent/fuzzing/output_mutated/original/code_round_0_with_passes.jsonl'
# output_data_path='/data/zlyuaj/muti-agent/fuzzing/output_mutated/original/code_round_0_with_passes_1.jsonl'
# x=[]
# with open(original_data_path,'r') as f:
#     x= [json.loads(line) for line in f]
# with open(output_data_path,'w+') as f:
#     for i in x[0]:
#         f.write(json.dumps(i) + '\n')
#     f.flush()
# original_data=[]
# passAt10_list=[]
# test_case_path= '/data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl'
# with open(test_case_path, 'r') as f:
#     test_cases = [json.loads(line) for line in f]

# with open(original_data_path, 'r') as f:
#     original_data = [json.loads(line) for line in f]
#     passes_list=[]
#     for i in range(len(original_data)):
#         # if original_data[i]['task_id']!='HumanEval/53':
#         #     continue
#         print("evaluating: "+ str(i))
#         original_data[i]['test'] = test_cases[i]['test_case_list']
#         score, passes, passAt1, passAt10,pass_results = evaluate_original(original_data[i],10)
#         print(passAt10,pass_results)
#         original_data[i]['pass_results']=pass_results
#         original_data[i]['passAt10']=passAt10
#         passAt10_list.append(passAt10)
#         # print(pass_results)
#         # print('-'*100)
#         # for xx in original_data[i]['test']:
#         #     print(xx)
#         # print(score,passes,passAt10)
# with open(output_data_path, 'a') as ff:
#     ff.write(json.dumps(original_data) + '\n')
#     ff.flush()
# print(passAt10_list)
    
        