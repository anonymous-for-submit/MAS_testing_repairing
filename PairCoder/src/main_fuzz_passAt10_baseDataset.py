import argparse
import ast
from gen.dataset_solver import solve_dataset,solve_dataset_one_problem
from log import get_logger, setup_logger
import logging
import datetime
import os
import json
from concurrent.futures import as_completed, ProcessPoolExecutor
from main_mutate import get_more_prompt_test
from evaluate_result import evaluate_one_MBPP,evaluate_one
logger = get_logger(__name__)
import sys
sys.path.append('/data/zlyuaj/muti-agent/PairCoder/')
sys.path.append('/data/zlyuaj/muti-agent/PairCoder/src/evaluate/')
sys.path.append('/data/zlyuaj/muti-agent/PairCoder/src/')
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="codecontest")
parser.add_argument("--split_name", type=str, default="valid")
parser.add_argument("--model", type=str, default="gpt-35-turbo")
parser.add_argument("--solution_file_name", type=str, default="solutions.json")
parser.add_argument("--id_range", default=None, nargs="+", type=int)
parser.add_argument('--id_list', default=None, type=str)
parser.add_argument('--dir_path', default=None, type=str)
parser.add_argument('--method', default='pair_programming')
parser.add_argument('--start_idx', type=int, default=-1)
parser.add_argument('--end_idx', type=int, default=1000)
parser.add_argument('--id_name', default='', type=str)



parser.add_argument('--output_path', type=str, default='output.jsonl')
parser.add_argument('--input_path', type=str, default='./data/HumanEval_test_case_ET.jsonl')
parser.add_argument('--output_file_name', type=str, default='test')
parser.add_argument('--num_generate', type=int, default=1)
parser.add_argument('--add_monitor', type=int, default=0)
parser.add_argument('--repair_plan', type=int, default=0)
parser.add_argument('--repair_code', type=int, default=0)
parser.add_argument('--run_multi_gen', type=int, default=0)
parser.add_argument('--repair_prompt_num', type=int, default=0)
args = parser.parse_args()
def run_solve_dataset(raw_id_list):
    
    if args.dir_path:
        print(f"current dir id: {args.dir_path}")
        dir_path = args.dir_path
    else:
        timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
        dir_path = f"{args.method}-{timestamp}"
        print(f"auto dir id: {args.dir_path}")
    
    return solve_dataset(args,
                    dataset_name=args.dataset,
                  split_name=args.split_name,
                  solution_file_name=args.solution_file_name,
                  num_iterations=args.num_generate,
                  id_range=args.id_range,
                  id_list=raw_id_list,
                  dir_path=dir_path,
                  method=args.method,
                  model = args.model,
                  repair_plan = bool(args.repair_plan),
                    repair_code = bool(args.repair_code),
                    add_monitor = bool(args.add_monitor),
                  )

if __name__ == "__main__":
    initial_output_path=args.output_path
    args.output_path=initial_output_path+'results-'+args.output_file_name+'/'
    x=2
    while os.path.exists(args.output_path):
        args.output_path=initial_output_path+'results-'+args.output_file_name+'_'+str(x)+'/'
        x+=1
    os.mkdir(args.output_path)
    print(args.output_path)
    print(args)

    # load dataset
    INPUTPATH=args.input_path
    loaded_dataset=[]
    with open(INPUTPATH, 'r') as f:
        # 导入输出
        loaded_dataset = [json.loads(line) for line in f]


    print('dataset number: ')
    print(len(loaded_dataset))


    # text, code, task_id, test_list, entry_point

    fail_list = []
    output_path = args.output_path + f'{args.dataset}.jsonl'
    

    with open(output_path, 'w+') as f:
        for idx,task in enumerate(loaded_dataset):
            if args.start_idx!=-1 and idx<args.start_idx:
                    continue
            if args.end_idx!=1000 and idx>args.end_idx:
                break
            if args.id_name!='' and task['name']!=args.id_name:
                continue
            # 返回 1. code 2. session_history 3. pass@10 4. pass_num
            print('-'*10+f'handling task {idx}'+'-'*10)

            if 'mbpp' in args.dataset or 'human' in args.dataset:
                if 'mbpp' in args.dataset:
                    task['name'] = 'mbpp/'+str(task['task_id'])
                    test_cases = task['test_list']
                if 'human' in args.dataset:
                    task['name'] = task['task_id']
                    test_cases = task['test_case_list']    
                task['description'] = task['prompt']

                def get_public_test(test_cases):
                    inputs,outputs=[],[]
                    for test in test_cases:
                        if '==' in test:
                            input_str, output_str=test.split('==')
                            l = input_str.find('(')
                            r = input_str.rfind(')')
                            input_str = input_str[l+1:r]
                            output_str = output_str.strip()
                            try:
                                input=eval(input_str)
                                if type(input)==tuple:
                                    input = list(input)
                                    for i in range(len(input)):
                                        if type(input[i])==tuple:
                                            input[i]=list(input[i])

                            except:
                                continue
                            inputs.append(str(input))
                            outputs.append(output_str)
                            break
                    return inputs,outputs


                inputs,outputs = get_public_test(test_cases)
                if not inputs:
                    inputs=['1']
                    outputs=['1']
                task['public_tests'] = {'input': inputs, 'is_valid_test': None, 'output': outputs}
                print(task['public_tests'] )
                task['private_tests'] = {'input': inputs, 'is_valid_test': None, 'output': outputs}
                task['generated_tests'] = {'input': [], 'is_valid_test': None, 'output': []}
                task['dataset_name'] = args.dataset
                # multi gen
                # {'input': ['[[3, 4, 5, 6], [5, 7, 4, 10]]', '[[1, 2, 3, 4], [5, 4, 3, 7]]', '[[11, 12, 14, 13], [17, 15, 14, 13]]'], 'is_valid_test': None, 'output': ['(4, 5)', '(3, 4)', '(13, 14)']}
                intent=task['description']
                repair_prompt=[]
                mutation_method = ['expand_one2two','condense_two2one','rephrase_one']
                if args.run_multi_gen==1:
                    for i in range(args.repair_prompt_num):
                        new_prompt= get_more_prompt_test(intent,args,mutation_method[i])
                        print('multi-gen-prompt:')
                        print(new_prompt)
                        repair_prompt.append(new_prompt)
                task['repair_prompt'] = [intent]+repair_prompt

                def form_prompt_mbpp(intent,task):
                    test_cases = task['test_list'][:1]
                    TEST_PROMPT = '\nexample:\n'
                    intent =  intent+TEST_PROMPT
                    for test in test_cases:
                        intent+=test[7:]+'\n'
                    return intent
                if 'mbpp' in args.dataset:
                    for i in range(len(task['repair_prompt'])):
                        task['repair_prompt'][i] = form_prompt_mbpp(task['repair_prompt'][i],task)
                
                
                
                [codes,plans,pass_At_10,pass_number] = solve_dataset_one_problem(problem = task,
                                                    dataset_name=args.dataset,
                                                    split_name=args.split_name,
                                                    solution_file_name=args.solution_file_name,
                                                    num_iterations=args.num_generate,
                                                    method=args.method,
                                                    model = args.model,
                                                    repair_plan = bool(args.repair_plan),
                                                    repair_code = bool(args.repair_code),
                                                    add_monitor = bool(args.add_monitor),
                                                    )
            else:
                [codes,plans,pass_At_10,pass_number,repair_prompt] = run_solve_dataset([idx])

            task['completions']=codes
            task['plans']=plans
            task['passed']=pass_At_10
            task['pass_num']=pass_number
            task['repair_prompt'] = repair_prompt

            if 'mbpp' in args.dataset:
                score, passes, passAt1, passAt10 = evaluate_one_MBPP(task,args.num_generate)
            if 'human' in args.dataset:
                score, passes, passAt1, passAt10 = evaluate_one(task,args.num_generate)

            task['passed']=passAt10
            task['pass_num']=passes

            f.write(json.dumps(task) + '\n')
            f.flush()
            # break



            




    