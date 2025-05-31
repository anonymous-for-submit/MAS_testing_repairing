import os
import copy
import json
import argparse
import tqdm
import numpy as np
import time
import random
import torch
import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset, load_from_disk
from main_mutate import mutate_one,mutate_one_nl,get_more_prompt_test
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from evaluate_result import evaluate_one_MBPP,evaluate_one
import multiprocessing

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="codecontest")
parser.add_argument("--split_name", type=str, default="valid")
parser.add_argument('--model', type=str, default='gpt-35-turbo')
parser.add_argument("--solution_file_name", type=str, default="solutions.json")
parser.add_argument("--id_range", default=None, nargs="+", type=int)
parser.add_argument('--id_list', default=None, type=str)
parser.add_argument('--dir_path', default=None, type=str)
parser.add_argument('--method', default='pair_programming')

parser.add_argument('--output_path', type=str, default='output.jsonl')
parser.add_argument('--input_path', type=str, default='./data/HumanEval_test_case_ET.jsonl')
parser.add_argument('--output_file_name', type=str, default='test')
parser.add_argument('--num_generate', type=int, default=1)
parser.add_argument('--parallel', type=int, default=0)

parser.add_argument('--mutate_method', type=str, default='random')
parser.add_argument('--num_round', type=int, default=3)

parser.add_argument('--save_seed', type=int , default=1)
parser.add_argument('--recover', type=int, default=0)
parser.add_argument('--calc_analyst', type=int, default=0)
parser.add_argument('--calc_final_result', type=int, default=1)
parser.add_argument('--save_all_seed', type=int, default=0)
parser.add_argument('--set_threshold_analyst', type=int, default=1)
parser.add_argument('--calc_relative_reward', type=int, default=1)
parser.add_argument('--clean_mutate_method', type=int, default=1)


parser.add_argument('--split_input', type=int, default=1)
parser.add_argument('--mutate_level', type=str, default='whole')

parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--beta', type=float, default=1)



parser.add_argument('--add_monitor', type=int, default=0)
parser.add_argument('--repair_plan', type=int, default=0)
parser.add_argument('--repair_code', type=int, default=0)
parser.add_argument('--run_multi_gen', type=int, default=0)
parser.add_argument('--repair_prompt_num', type=int, default=0)

args = parser.parse_args()


def record(args,prompt_nodes,initial_seed_num):
    print('in recording...')
    print('len prompt_nodes')
    print(len(prompt_nodes))
    print('initial_seed_num')
    print(initial_seed_num)
    print([prompt_node.finish for prompt_node in prompt_nodes[:initial_seed_num]])
    total_passAt10=[prompt_node.finish for prompt_node in prompt_nodes[:initial_seed_num]].count(False)
    print('total pass@10:' + str(total_passAt10))

    final_reuslt_output_path=args.output_path+'_final_result.jsonl'
    print('-'*30)
    print('saving result into: '+final_reuslt_output_path)
    with open(final_reuslt_output_path, 'a') as f:
        f.write(str(total_passAt10)+'\n')
        f.flush()
    return total_passAt10


def save_node(args,prompt_nodes,initial_seed_num,round):
    node_output_path=args.output_path+'_node_{}.jsonl'.format(round)
    prompt_nodes2save=prompt_nodes[initial_seed_num:]
    print('-'*30)
    print('saving node into: '+node_output_path)
    with open(node_output_path, 'w+') as f:
        for prompt_node in prompt_nodes2save:
            result={
                'visited_num': prompt_node.visited_num,
                'score':prompt_node.score,
                'passes':prompt_node.passes,
                'reward_score':prompt_node.reward_score,
                'finish':prompt_node.finish,
                'level':prompt_node.level,
                'index':prompt_node.index,
                'parent':prompt_node.parent.index,
                'child':[child.index for child in prompt_node.child],
                'solution':prompt_node.solution
            }
            f.write(json.dumps(result) + '\n')
            f.flush()



from code_contests.data.provider import CodeContestDataProvider
from gen.coding_competitor import CodeContestsCompetitor
from gen.utils import evaluate_solution_on_subset
from log import setup_logger
from settings.config_loader import get_settings
from gen.dataset_solver import solve_dataset,solve_dataset_one_problem,solve_dataset_one_problem_parallel
import toml
import datetime

if __name__ == '__main__':

    
    
    # initial_output_path=args.output_path
    # args.output_path=initial_output_path+'results-'+args.output_file_name+'/'
    # x=2
    # while os.path.exists(args.output_path):
    #     args.output_path=initial_output_path+'results-'+args.output_file_name+'_'+str(x)+'/'
    #     x+=1
    # os.mkdir(args.output_path)
    # print(args.output_path)
    print(args)

    from datasets import load_dataset
    # ds = load_dataset("dz1/CodeScore-MBPP-ET")



    data_provider = CodeContestDataProvider(dataset_location=args.dataset)
    num_problems = len(data_provider.dataset[args.split_name])
    print(f'num_problems: {num_problems}')

    # load dataset
    INPUTPATH=args.input_path
    
    prompt_nodes=[]
    with open(INPUTPATH, 'r') as f:
        # 导入输出
        prompt_nodes = [json.loads(line) for line in f]

    comletions_plans_passes = []
    for prompt_node in prompt_nodes:
        if not prompt_node['score']:
            comletions_plans_passes.append(prompt_node['solution'])

       


    



    private_tests_dict = {}
    loaded_dataset=[]
    for problem_number in range(num_problems):
        problem_name = data_provider.dataset[args.split_name][int(problem_number)]['name']

        # 从导入的dataset里面找到问题
        problem = data_provider.find_problem(ds=data_provider.dataset, problem_name=problem_name, split_name=args.split_name)
        problem['dataset_name'] = args.dataset
        loaded_dataset.append(problem)

    private_tests_dict = {data['name']:data['private_tests'] for data in loaded_dataset} 

    loaded_dataset = comletions_plans_passes 
    

    # # assert len(comletions_plans_passes) == len(loaded_dataset)

    # data_dict = {data['name']:data for data in loaded_dataset}
    # for 
    # for i in range(len(loaded_dataset)):
    #     data = data_dict[loaded_dataset[i]['name']]
    #     loaded_dataset[i]['completions']=data['completions']
    #     loaded_dataset[i]['plans']=data['plans']
    #     loaded_dataset[i]["passed"]=data["passed"]
    #     loaded_dataset[i]["pass_num"]=data['pass_num']
    #     if 'human' in args.dataset:
    #         loaded_dataset[i]['nl']=data['nl']
    #         loaded_dataset[i]["func"]=data["func"]
    #         loaded_dataset[i]["examples"]=data['examples']

    # print(loaded_dataset[0].keys())

    

    for data in loaded_dataset:
        if 'private_tests' in data.keys():
            data.pop('private_tests')

    # print('-'*40)
    # for key,value in private_tests_dict.items():
    #     print('-'*40)
    #     print(key)
    #     print(value)
    #     break

    print(f'private_tests_dict {len(private_tests_dict)}')


    
        
    


    prompt_nodes=[]
    passAt10s=[]

    initial_seed = loaded_dataset

    

    initial_seed_num = len(loaded_dataset)
    print(f'len of loaded problems: {initial_seed_num}')

    multi_gen_tag=''
    repair_plan_tag=''
    repair_code_tag=''
    name_tag='1-1'
    if args.run_multi_gen==1:
        multi_gen_tag = '_repair_prompt_num_{}'.format(str(args.repair_prompt_num))
    if args.repair_plan==1:
        repair_plan_tag = '_repair_plan'
    if args.repair_code==1:
        repair_code_tag = '_repair_code'


    import time
    t=0
    output_completions_file = '/data/zlyuaj/muti-agent/PairCoder/z_scripts/repair/time_analysis/mix_{}_{}_{}{}{}{}.jsonl'.format(args.dataset,args.model,name_tag,multi_gen_tag,repair_plan_tag,repair_code_tag)
    with open(output_completions_file,'w+') as f:
        loaded_dataset=loaded_dataset[:10]
        for idx,task in enumerate(loaded_dataset):
            # if idx>=22:
            #     continue
            


            intent=task['description']

            s=time.time()


            repair_prompt = []
            mutation_method = ['expand_one2two','condense_two2one','rephrase_one']
            if args.run_multi_gen==1:
                for i in range(args.repair_prompt_num):
                    new_prompt= get_more_prompt_test(intent,args,mutation_method[i])
                    print('multi-gen-prompt:')
                    print(new_prompt)
                    repair_prompt.append(new_prompt)
            task['repair_prompt'] = [intent]+repair_prompt


            score, passes=-1,-1
            # task['private_tests'] = private_tests_dict[task['name']]


            # test_cases=''
            # if 'mbpp' in args.dataset:
            #     # print(task.keys())
            #     test_cases = task['test_list']
            # # if 'human' in args.dataset:
            # #     test_cases = task['test_case_list']    
            # test_cases = ['a==b']
            # task['description'] = task['repair_prompt'][0]

            # def get_public_test(test_cases):
            #         inputs,outputs=[],[]
            #         for test in test_cases:
            #             if '==' in test:
            #                 input_str, output_str=test.split('==')
            #                 l = input_str.find('(')
            #                 r = input_str.rfind(')')
            #                 input_str = input_str[l+1:r]
            #                 output_str = output_str.strip()
            #                 try:
            #                     input=eval(input_str)
            #                     if type(input)==tuple:
            #                         input = list(input)
            #                         for i in range(len(input)):
            #                             if type(input[i])==tuple:
            #                                 input[i]=list(input[i])

            #                 except:
            #                     continue
            #                 inputs.append(str(input))
            #                 outputs.append(output_str)
            #                 break
            #         return inputs,outputs


            # inputs,outputs = get_public_test(test_cases)
            # if not inputs:
            #     inputs=['1']
            #     outputs=['1']
            # task['public_tests'] = {'input': inputs, 'is_valid_test': None, 'output': outputs}
            # print(task['public_tests'] )
            # task['private_tests'] = {'input': inputs, 'is_valid_test': None, 'output': outputs}
            # task['generated_tests'] = {'input': [], 'is_valid_test': None, 'output': []}
            task['dataset_name'] = args.dataset
            # 得到code，plan，pass的所有信息
            passing_results=[]
            codes,plans,pass_At_10,pass_number = [],[],False,0
            if not bool(args.parallel):
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
            e=time.time()
        

                
            # print('-'*10)
            # print(codes)
            # print('-'*10)
            # print(plans)
            # print('-'*10)
            # print(pass_At_10)
            # print('-'*10)
            # print(pass_number)

            # task.pop('private_tests')

            # task['completions_after_repair']=codes
            # task['plans_after_repair']=plans

            # if 'mbpp' in args.dataset:
            #     score, passes, passAt1, passAt10 = evaluate_one_MBPP(task,args.num_generate)
            #     pass_At_10,pass_number = passAt10,passes
            # if 'human' in args.dataset:
            #     score, passes, passAt1, passAt10 = evaluate_one(task,args.num_generate)
            #     pass_At_10,pass_number = passAt10,passes
            
            # task['pass_after_repair'] = pass_At_10
            # task['pass_num_after_repair'] = pass_number
            # task['round_in_repair']=idx
            c=e-s
            t+=c
            
            
            f.write(str(c) + '\n')
            f.flush()
        f.write("-"*100 + '\n')
        f.write(str(t) + '\n')
        f.write(str(t/10) + '\n')
        f.flush()
        



            

        

        # print('len(prompt_nodes): '+str(len(prompt_nodes)))
        # print('reward before fuzzing')
        # # print(select_policy.rewards)
        
        # # select_policy.update([mutated_seed],prompt_nodes)
        # select_policy.rewards[seed.index]=seed.reward_score
        # # print('reward after fuzzing')
        # print([prompt_node.finish for prompt_node in prompt_nodes])
        # print(select_policy.rewards)

        
        



    

        
    

