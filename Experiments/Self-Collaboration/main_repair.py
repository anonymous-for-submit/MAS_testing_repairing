import openai
from openai import OpenAI
from openai import AzureOpenAI
# client = OpenAI(
#     # 输入转发API Key
#     api_key="sk-NsLLS6Bbm06SDgbx3BJkyHsEys50pj9TqlZB7PrIJHFSIzmI",
#     base_url="https://api.chatanywhere.com.cn/v1"
# )
import time
from main_mutate import choose_code,get_more_prompt_test

import pandas as pd
import json
import os,sys
# print(sys.path)
# print(sys.path)
from session import Session
from datasets import load_dataset, load_from_disk
from utils import prompt_split_humaneval, find_method_name, code_split, build_test_method
from evaluate_result import evaluate_all,evaluate_one,evaluate_one_MBPP,evaluate_one_codecontest
from concurrent.futures import as_completed, ProcessPoolExecutor
import multiprocessing
import os
from LLMCodeChoice.src.llm_evidence.ogis import run_humanEval_choose_completions

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_generate', type=int, default=10)
parser.add_argument('--model', type=str, default='gpt-35-turbo')
parser.add_argument('--testing', type=int, default=0)
parser.add_argument('--calc_pass_at_1', type=int, default=1)
parser.add_argument('--repair_plan', type=int, default=0)
parser.add_argument('--add_monitor', type=int, default=0)
parser.add_argument('--run_multi_gen', type=int, default=0)
parser.add_argument('--repair_code', type=int, default=0)
parser.add_argument('--repair_prompt_num', type=int, default=0)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--dataset', type=str, default='humaneval')
parser.add_argument('--MBPP_test_case_num', type=int, default=1)
parser.add_argument('--file_name', type=str, default='')
parser.add_argument('--start_idx', type=int, default=0)


parser.add_argument('--data_path', type=str, default='/data/zlyuaj/muti-agent/fuzzing/output_fuzzing_one_per_time/pass@10/split_input/sentence_level/alabation_mutate_method/')

parser.add_argument('--select_completion_ogis', type=int, default=0)
parser.add_argument('--code_index', type=int, default=1000)

args = parser.parse_args()

model = args.model
num_generate  = args.num_generate
repair_prompt_num = args.repair_prompt_num
data_path = args.data_path
names=[]
if not args.file_name:
    names=os.listdir(data_path)
    names = names[:1]
else:
    names=[args.file_name]
repair_nums = []
repair_nums_passAt1=[]
new_repair_nums_passAt1=[]

total_nums = []
invalid_cnt=0
# if args.testing==1:
#     num_generate=1
#     names=names[:1]
#     args.repair_prompt_list=0
# # names = names[:1]
#     repair_nums = []


print(names)
total_nums = []
multi_gen_tag=''
repair_plan_tag=''
repair_code_tag=''
name_tag=''
if args.run_multi_gen==1:
    multi_gen_tag = '_repair_prompt_num_{}'.format(str(args.repair_prompt_num))
if args.repair_plan==1:
    repair_plan_tag = '_repair_plan'
if args.repair_code==1:
    repair_code_tag = '_repair_code'

for name in names:
    # if '1-1' not in name or 'repair' in name:
    #     continue
    cur_data_path = data_path
    cur_data_path = cur_data_path+name
    # i=1000
    cur_data_path=cur_data_path+'/_node_{}.jsonl'.format(args.code_index)
    # print(cur_data_path)

    no_pass=[]
    original_dataset=[]
    task_dict={}
    if 'human' in args.dataset:
        original_data_path ='/data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl'
    elif 'mbpp' in args.dataset:
        original_data_path = '/data/zlyuaj/muti-agent/fuzzing/data/MBPP_ET.jsonl'
    elif 'contest' in args.dataset:
        original_data_path = '/data/zlyuaj/muti-agent/fuzzing/data/CodeContest_Test.jsonl'
    with open(original_data_path, 'r') as f:
        # 导入输出
        original_dataset = [json.loads(line) for line in f]
        task_dict = {task['task_id']:task for task in original_dataset} 
    with open(cur_data_path, 'r') as f:
        datas = [json.loads(line) for line in f]
        for data in datas:
            # if 'score' not in data.keys():
            #     print(data)
            if 'human' in args.dataset: 
                task_id=int(data['task_id'].split('/')[-1])
                if task_id==113 or task_id==78:
                    continue
                data['test_case_list']=task_dict[data['task_id']]['test_case_list']
            if 'mbpp' in args.dataset:
                data['test_list']=task_dict[data['task_id']]['test_list']
            if 'codecontest' in args.dataset:
                data['test_list']=task_dict[data['task_id']]['test_list']
            if not data['score']:
                no_pass.append(data)
    print('len of no pass data:' + str(len(no_pass)))


    repair_num=0
    total_passAt1=0
    new_total_passAt1=0
    total_num=len(no_pass)
    # print(total_num)

    from roles.rule_descriptions_actc import TEAM, ANALYST, PYTHON_DEVELOPER, TESTER
    if '1-1' in name:
        name_tag = '1-1'
    elif 'by_pass' in name:
        name_tag = 'by_pass'
    elif 'no_save_seed' in name:
        name_tag = 'no_seed'
    else:
        name_tag=name
    output_completions_file = 'mix_{}_{}{}{}{}.jsonl'.format(model,name_tag,multi_gen_tag,repair_plan_tag,repair_code_tag)
    with open(output_completions_file,'w+') as f:
        for idx,no_pass_data in enumerate(no_pass):
            if idx<args.start_idx:
                continue
            # if 'fib(' not in no_pass_data['prompt']:
            #     continue
            print('evaluating data idx: {}, task id: {}'.format(idx,no_pass_data['task_id']))

            score, passes=-1,-1
            task=no_pass_data

            
            
            codes=[]
            session_historys=[]
            intent=task['prompt']

            print('prompt:')
            print(intent)
            print('-'*30)
            
            repair_prompt = []
            mutation_method = ['expand_one2two','condense_two2one','rephrase_one']
            if args.run_multi_gen==1:
                for i in range(repair_prompt_num):
                    new_prompt= get_more_prompt_test(intent,args,mutation_method[i])
                    print('multi-gen-prompt:')
                    print(new_prompt)
                    if not new_prompt:
                        continue
                    repair_prompt.append(new_prompt)
            task['repair_prompt'] = [intent]+repair_prompt
            before_func = ''


            if 'human' in args.dataset:
                task_id=int(no_pass_data['task_id'].split('/')[-1])
                method_name=''
                for original_data in original_dataset:
                    cur_task_id=int(original_data['task_id'].split('/')[-1])
                    if cur_task_id==task_id:
                        method_name = original_data['entry_point']
                before_func = prompt_split_humaneval(task['prompt'],method_name)
            elif 'mbpp' in args.dataset:
                task_id=int(no_pass_data['task_id'])
                original_task = task_dict[no_pass_data['task_id']]
                method_name = original_task['entry_point']
                
                task['entry_point'] = original_task['entry_point']
                # print(test_cases)
                def form_prompt_mbpp(intent,task):
                    test_cases = task['test_list'][:args.MBPP_test_case_num]
                    TEST_PROMPT = '\nexample:\n'
                    intent =  intent+TEST_PROMPT
                    for test in test_cases:
                        intent+=test[7:]+'\n'
                    return intent
                for i in range(len(task['repair_prompt'])):
                    task['repair_prompt'][i] = form_prompt_mbpp(task['repair_prompt'][i],original_task)
            elif 'contest' in args.dataset:
                task_id=int(no_pass_data['task_id'].split('/')[-1])
                original_task = task_dict[no_pass_data['task_id']]
                def form_prompt_codecontest(intent):
                    prompt = f"{intent}\n\n-------\nImportant Note: You must follow the input output format. Input must be taken from standard input and output must be given to standard output. The code will be tested against multiple test cases and all the test cases must be passed."
                    if 'deepseek' in args.model:
                        prompt +=  '''\nWrite a main() function and use input() function to read input from stdin'''
                        p="\nPlease use 'if __name__ == '__main__':' and 'input()' function to read input\n"
                    return prompt
                for i in range(len(task['repair_prompt'])):
                    task['repair_prompt'][i] = form_prompt_codecontest(task['repair_prompt'][i])
                # task['test_list'] = original_task[]
            

            try:
                # 进行分工流程在这里，输入了prompt为intent
                futures = []

                
                
                # print('safeffa')
                # print(len(mutated_seed.solution['repair_prompt']))
                with ProcessPoolExecutor() as executor:
                    for cnt in range(args.num_generate):
                        
                        # print(len(task['repair_prompt']))
                        ii = (cnt//len(task['repair_prompt']))%len(task['repair_prompt'])
                        # print(ii)
                        session = Session(TEAM, ANALYST, PYTHON_DEVELOPER, TESTER,requirement=task['repair_prompt'][ii], model=model, repair_prompt=bool(args.repair_plan),repair_code=bool(args.repair_code),add_monitor = bool(args.add_monitor), majority=1, 
                                    max_tokens=512, temperature=0.0, 
                                    top_p=0.95, max_round=2, before_func=before_func)
                        future= executor.submit(session.run_session,0,0)
                    
                        futures.append(future)
                all_code=[]   
                all_session_historys = []
                for cnt, future in enumerate(as_completed(futures)):
                    # print(future.result())
                    code, session_history, need_second_round, finally_pass=future.result()
                    # if cnt%(1+args.repair_prompt_num)==0:
                    #     session_historys.append(session_history)
                    if 'deepseek' in args.model and 'contest' in args.dataset:
                        code+='\nmain()'
                    codes.append(code)
                    session_historys.append(session_history)
            except RuntimeError as e:
                print(str(e))
                print("task-%d fail"%(task['task_id']))
                continue

            print('current data idx: '+str(idx))
            no_pass_data['completions_after_repair']=codes
            # no_pass_data['completions']=codes
            no_pass_data['plan_after_repair']=session_historys
            

            if 'human' in args.dataset:
                score, passes,passAt1, passAt10= evaluate_one(no_pass_data,len(no_pass_data['completions']))
            elif 'mbpp' in args.dataset:
                score, passes,passAt1, passAt10= evaluate_one_MBPP(no_pass_data,len(no_pass_data['completions']))
            elif 'contest' in args.dataset:
                score, passes,passAt1, passAt10= evaluate_one_codecontest(no_pass_data,len(no_pass_data['completions']))
            no_pass_data['pass_after_repair']=passes
            no_pass_data['pass@10_after_repair']=passAt10
            no_pass_data['pass@1_after_repair']=passes/len(no_pass_data['completions'])
            total_passAt1+=passes/len(no_pass_data['completions'])
            new_total_passAt1+=passes/len(no_pass_data['completions'])
            no_pass_data['pass@1_after_repair_ogis']=passes/len(no_pass_data['completions'])
            if score ==-1:
                invalid_cnt+=1
                print('invalid (jailbreak or other issue filter by azure)')
                print('invalid_cnt: '+str(invalid_cnt))
            print(passAt10)
            total_passAt1+=passes/len(no_pass_data['completions'])
            f.write(json.dumps(task) + '\n')
            f.flush()


            if passAt10:
                repair_num+=1
        print(name)
        print('repair_num: ' + str(repair_num))
        print('total_passAt1: ' + str(total_passAt1))
        print('new_total_passAt1: ' + str(new_total_passAt1))
        
        print('total_num: ' + str(total_num))
        print('repair_num/total_num: ' + str(repair_num/total_num))

        repair_nums.append(repair_num)
        total_nums.append(total_num)
        repair_nums_passAt1.append(total_passAt1)
        new_repair_nums_passAt1.append(new_total_passAt1)

        
        
        if args.testing==1:
            break
    if args.testing==1:
        break    
output_file = 'mix_{}_{}{}{}.txt'.format(model,multi_gen_tag,repair_plan_tag,repair_code_tag)
with open(output_file,'w+') as f:
    for i in range(len(names)):
        f.write(names[i]+'\n')
        f.write('repair_nums_pass@10: '+str(repair_nums[i])+'\n')
        f.write('repair_nums_pass@1: '+str(repair_nums_passAt1[i])+'\n')
        f.write('new_repair_nums_pass@1: '+str(new_repair_nums_passAt1[i])+'\n')
        f.write('total_nums@: '+str(total_nums[i])+'\n')
'''
add one 26/44
random no seed 18/38
random by pass 20/36
'''