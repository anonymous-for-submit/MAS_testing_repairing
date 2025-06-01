
# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path

import agentops
import typer

from const import CONFIG_ROOT
from metagpt.utils.project_repo import ProjectRepo
import os
import copy
import json
import argparse
import tqdm
import numpy as np
import time
import random
# import torch
# import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset, load_from_disk
from collections import defaultdict
from evaluate_result import evaluate_one
from concurrent.futures import as_completed, ProcessPoolExecutor
import multiprocessing
from main_mutate  import extract_code_from_sourse
from _utils import prompt_split_humaneval
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='humaneval')
parser.add_argument('--lang', type=str, default='python')
parser.add_argument('--output_path', type=str, default='output/')
parser.add_argument('--input_path', type=str, default='data/HumanEval_test_case_ET.jsonl')
parser.add_argument('--mutate_method', type=str, default='random')
parser.add_argument('--dataset_type', type=str, default='HumanEval')
parser.add_argument('--output_file_name', type=str, default='test')
parser.add_argument('--num_round', type=int, default=3)
parser.add_argument('--num_generate', type=int, default=1)
parser.add_argument('--do_fuzz', type=bool, default=True)
parser.add_argument('--save_seed', type=int , default=1)
parser.add_argument('--recover', type=int, default=0)
parser.add_argument('--reward_ratio', type=int, default=1)
parser.add_argument('--calc_analyst', type=int, default=0)
parser.add_argument('--calc_original_plan', type=int, default=0)
parser.add_argument('--calc_final_result', type=int, default=1)
parser.add_argument('--save_all_seed', type=int, default=0)
parser.add_argument('--clean_data', type=int, default=1)
parser.add_argument('--set_threshold_analyst', type=int, default=1)
parser.add_argument('--calc_relative_reward', type=int, default=1)
parser.add_argument('--calc_llm_absolute', type=int, default=0)
parser.add_argument('--single_llm', type=int, default=0)
parser.add_argument('--clean_mutate_method', type=int, default=0)

parser.add_argument('--parallel', type=int, default=0)
parser.add_argument('--split_input', type=int, default=0)
parser.add_argument('--mutate_level', type=str, default='whole')
parser.add_argument('--llm_critic', type=str, default='none')
parser.add_argument('--with_reference', type=int, default=1)

parser.add_argument('--only_consider_passed_cases', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--beta', type=float, default=1)


parser.add_argument('--signature', action='store_true')
parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0301')
parser.add_argument('--max_round', type=int, default=2)

parser.add_argument('--max_tokens', type=int, default=512) 
parser.add_argument('--majority', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--top_p', type=float, default=0.95)

parser.add_argument('--fail_list', type=list, default=[])
parser.add_argument('--append', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument("--timeout", type=float, default=10, help="how many seconds to wait during execution for each test case")
args = parser.parse_args()
def generate_repo(
    idea,
    investment=3.0,
    n_round=5,
    code_review=True,
    run_tests=False,
    implement=True,
    project_name="",
    inc=False,
    project_path="",
    reqa_file="",
    max_auto_summarize_code=0,
    recover_path=None,
) -> ProjectRepo:
    """Run the startup logic. Can be called from CLI or other Python scripts."""
    from metagpt.config2 import config
    from metagpt.context import Context
    from metagpt.roles import (
        Architect,
        Engineer,
        ProductManager,
        ProjectManager,
        QaEngineer,
    )
    from metagpt.team import Team
    # print (config)
    # if config.agentops_api_key != "":
    #     agentops.init(config.agentops_api_key, tags=["software_company"])

    config.update_via_cli(project_path, project_name, inc, reqa_file, max_auto_summarize_code)
    ctx = Context(config=config)

    if not recover_path:
        # 建立公司，并招募员工
        company = Team(context=ctx)
        # 先找三个员工
        company.hire(
            [
                ProductManager(),
                Architect(),
                ProjectManager(),
            ]
        )

        if implement or code_review:
            company.hire([Engineer(n_borg=5, use_code_review=code_review)])

        if run_tests:
            company.hire([QaEngineer()])
    else:
        stg_path = Path(recover_path)
        if not stg_path.exists() or not str(stg_path).endswith("team"):
            raise FileNotFoundError(f"{recover_path} not exists or not endswith `team`")

        company = Team.deserialize(stg_path=stg_path, context=ctx)
        idea = company.idea
    # 做项目评估，仅从budget角度

    company.invest(investment)
    # 根据输入的idea进行软件开发
    company.run_project(idea)
    asyncio.run(company.run(n_round=n_round))

    # if config.agentops_api_key != "":
    #     agentops.end_session("Success")

    return ctx.repo


def startup(
    idea: str = 'write a python function to count 1-100',
    investment: float = 3.0,
    n_round: int = 5,
    code_review: bool = True,
    run_tests: bool = False,
    implement: bool = True,
    project_name: str = "",
    inc: bool = False,
    project_path: str = "",
    reqa_file: str ="",
    max_auto_summarize_code: int = 0,
    recover_path: str = None,
    init_config: bool = False,
):
    """Run a startup. Be a boss."""

    if idea is None:
        typer.echo("Missing argument 'IDEA'. Run 'metagpt --help' for more information.")
        raise typer.Exit()

    return generate_repo(
        idea,
        investment,
        n_round,
        code_review,
        run_tests,
        implement,
        project_name,
        inc,
        project_path,
        reqa_file,
        max_auto_summarize_code,
        recover_path,
    )

def extract_code_from_repo(file_path,prompt,method_name,code_in_prompt):
    files=os.listdir(file_path)
    num_py_files = len(files)
    # print(files)
    file_name = files[0]
    if 'main' in file_name and num_py_files>1:
        file_name = files[1]
    # print(file_name)
    sourse=''
    code=''
    with open(file_path+ '/'+file_name,'r') as f:
        sourse = f.read()
        # print(sourse)
        # code =''
        code  = extract_code_from_sourse(sourse,code_in_prompt)
    return code,sourse
def extract_plan_from_repo(file_path):
    prd_path = file_path +'/docs/prd'
    system_design_path = file_path +'/docs/system_design'
    if not os.path.exists(prd_path):
        RequirementAnalysis,RequirementPool='',''
    else:
        path = prd_path + '/'+os.listdir(prd_path)[0]
        with open(path,'r') as f:
            prd=json.load(f)
            RequirementAnalysis = prd['Requirement Analysis']
            RequirementPool = prd['Requirement Pool']
    if not os.path.exists(system_design_path):
        ImplementationApproach=''
    else:
        path = system_design_path + '/'+os.listdir(system_design_path)[0]
        with open(path,'r') as f:
            system_design=json.load(f)
            ImplementationApproach = system_design['Implementation approach']
    return RequirementAnalysis,RequirementPool,ImplementationApproach
def delete_repo(file_path):
    import shutil
    shutil.rmtree(file_path) 

if __name__ == '__main__':
    coding_prompt='write ONLY ONE static python function for the requirement. Make sure that your output only have one static function that solve the input requirement. Here is the input requirement: \n \n'
    
    
    
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
    

    # loaded_dataset[:3]

    print(len(loaded_dataset))
    prompt_nodes=[]
    passAt10s=[]

    initial_seed = loaded_dataset
    initial_seed_num=len(loaded_dataset)


    # text, code, task_id, test_list, entry_point

    fail_list = []
    output_path = args.output_path + 'HumanEval_ET.jsonl'
    output_codes=[]
    with open(output_path, 'w+') as f:
        args.num_generate=10
        import csv
        with open("sourse_code.csv","a",newline='') as csvfile:
            writer = csv.writer(csvfile)
            for idx,task in enumerate(loaded_dataset):
                # if idx>0:
                #     break
                print('-'*10+'executing task: {}'.format(idx)+'-'*10)
                codes = ['']*args.num_generate
                all_RequirementAnalysis,all_RequirementPool,all_ImplementationApproach = ['']*args.num_generate,['']*args.num_generate,['']*args.num_generate
                session_historys=[]
                intent=task['prompt']
                method_name = task['entry_point']
                before_func,code_in_prompt = prompt_split_humaneval(intent,method_name)
                # if before_func:
                    # print(before_func)
                # print(before_func,code)
                # new_loop = asyncio.new_event_loop()
                # asyncio.set_event_loop(new_loop)
                # repo=startup(idea=coding_prompt+intent,project_name=method_name)
                # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ method_name+'/'+method_name
                # code = extract_code_from_repo(file_path,intent)
                # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ method_name
                # delete_repo(file_path)
                # print(code)
                # output_codes.append(code)
                code_in_prompt = code_in_prompt.split('\n')[0]
                # if 'make_palindrome' not in code_in_prompt:
                #     continue
                # print(code_in_prompt)
                # regenerate=[]
                futures=[]
                with ProcessPoolExecutor() as executor:
                    for cnt in range(10):
                        new_method_name = method_name+"_"+str(cnt)
                        file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name+'/'+new_method_name
                        if not os.path.exists(file_path):
                            # regenerate.append(cnt)
                            continue
                        # code,sourse = extract_code_from_repo(file_path,intent,method_name,code_in_prompt)
                        future= executor.submit(extract_code_from_repo,file_path,intent,method_name,code_in_prompt)
                        futures.append(future)
                
                    
                    
                    
                for cnt, future in enumerate(as_completed(futures)):        
                    code,sourse = future.result()
                    code=before_func+code
                    writer.writerow([code,sourse])
                    # print(code)
                    file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name
                    RequirementAnalysis,RequirementPool,ImplementationApproach = extract_plan_from_repo(file_path)
                                # print(RequirementAnalysis,RequirementPool,ImplementationApproach)
                                # delete_repo(file_path)
                                # code, session_history=future.result()
                                # print('#'*30)
                                # print(code)
                    codes[cnt]=code
                    all_RequirementAnalysis[cnt] = RequirementAnalysis
                    all_RequirementPool[cnt] = RequirementPool
                    all_ImplementationApproach[cnt] = ImplementationApproach
                # break
                task['completions']=codes
                task['all_RequirementAnalysis']=all_RequirementAnalysis
                task['all_RequirementPool']=all_RequirementPool
                task['all_ImplementationApproach']=all_ImplementationApproach
                
                score, passes,passAt1, passAt10= evaluate_one(task,args.num_generate)
                # mutated_seed.score=score
                
                print(passAt10)
                task['pass'] = passAt10
                task['pass_num'] = passes
                task['round']=idx
                # entry_point = find_method_name(code)

                passAt10s.append(passAt10)
                f.write(json.dumps(task) + '\n')
                f.flush()
                if idx%10==0:
                    print('current round: '+str(idx))
                    print('current pass@10: '+ str(passAt10s.count(True)/len(passAt10s)))
            csvfile.close()



            # skip

    #         def generate(method_name,intent,generate_ids,all_RequirementAnalysis,all_RequirementPool,all_ImplementationApproach):
    #             try:
    #                 # 进行分工流程在这里，输入了prompt为intent
    #                 futures=[]
    #                 regenerate = []
    #                 split_para=1
    #                 for __ in range(split_para):
    #                     with ProcessPoolExecutor() as executor:
    #                         for cnt in generate_ids:
    #                             new_loop = asyncio.new_event_loop()
    #                             asyncio.set_event_loop(new_loop)
    #                             new_method_name = method_name+"_"+str(cnt)
    #                             # repo=startup(idea=coding_prompt+intent,project_name=method_name)
    #                             # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name+'/'+new_method_name
    #                             # code = extract_code_from_repo(file_path,intent)
    #                             # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name
    #                             # delete_repo(file_path)
    #                             # future= executor.submit(session.run_session,need_second_round,finally_pass)
    #                             future= executor.submit(startup,idea=coding_prompt+intent,project_name=new_method_name)
                            
    #                             futures.append(future)
    #                     for cnt, future in enumerate(as_completed(futures)):
    #                         # print(future.result())
    #                         new_method_name = method_name+"_"+str(cnt)
    #                         file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name+'/'+new_method_name
    #                         if not os.path.exists(file_path):
    #                             regenerate.append(cnt)
    #                             continue
    #                         code = extract_code_from_repo(file_path,intent)
    #                         file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name
    #                         RequirementAnalysis,RequirementPool,ImplementationApproach = extract_plan_from_repo(file_path)
    #                         # print(RequirementAnalysis,RequirementPool,ImplementationApproach)
    #                         # delete_repo(file_path)
    #                         # code, session_history=future.result()
    #                         # print('#'*30)
    #                         # print(code)
    #                         codes[cnt]=code
    #                         all_RequirementAnalysis[cnt] = RequirementAnalysis
    #                         all_RequirementPool[cnt] = RequirementPool
    #                         all_ImplementationApproach[cnt] = ImplementationApproach
    #                         # output_codes.append(code)
    #                         # print(code)
    #                 return codes,all_RequirementAnalysis,all_RequirementPool,all_ImplementationApproach, regenerate
    #                         # session_historys.append(session_history)
    #             except Exception as e:
    #                 print(str(e))
    #                 print("task-%s fail"%(task['task_id']))
    #                 fail_list.append(task['task_id'])
    #                 return codes,all_RequirementAnalysis,all_RequirementPool,all_ImplementationApproach, generate_ids
                
    #         generate_ids=[i for i in range(1,args.num_generate+1)]
    #         while len(generate_ids)>0:
    #             codes,all_RequirementAnalysis,all_RequirementPool,all_ImplementationApproach,generate_ids = generate(method_name,intent,generate_ids,all_RequirementAnalysis,all_RequirementPool,all_ImplementationApproach)

    #         # codes.append(code)
    #         task['completions']=codes
    #         task['all_RequirementAnalysis']=all_RequirementAnalysis
    #         task['all_RequirementPool']=all_RequirementPool
    #         task['all_ImplementationApproach']=all_ImplementationApproach
    #         print('evaluating ...')
    #         score, passes,passAt1, passAt10= evaluate_one(task,args.num_generate)
    #         # mutated_seed.score=score

            
    #         print(passAt10)
    #         task['pass'] = passAt10
    #         task['pass_num'] = passes
    #         task['round']=idx
    #         # entry_point = find_method_name(code)

    #         passAt10s.append(passAt10)
    #         f.write(json.dumps(task) + '\n')
    #         f.flush()
    #         if idx%10==0:
    #             print('current round: '+str(idx))
    #             print('current pass@10: '+ str(passAt10s.count(True)/len(passAt10s)))
    # print('-'*100)
    # print('final_result: '+str(passAt10s.count(True)/len(passAt10s)))
           



