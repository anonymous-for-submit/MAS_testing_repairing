
# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path

import agentops
import typer
import sys
sys.path.append('/home/zlyuaj/muti-agent/MetaGPT')
sys.path.append('/home/zlyuaj/muti-agent/MetaGPT/metagpt')
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
import torch
import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset, load_from_disk
from collections import defaultdict
from evaluate_result import evaluate_one,evaluate_one_codecontest,evaluate_one_MBPP

from concurrent.futures import as_completed, ProcessPoolExecutor
from main_mutate import mutate_one,mutate_one_nl,get_more_prompt_test
import multiprocessing
# from main_mutate  import extract_code_from_sourse
from _utils import prompt_split_humaneval
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='humaneval')
parser.add_argument('--output_path', type=str, default='output/')
parser.add_argument('--input_path', type=str, default='data/HumanEval_test_case_ET.jsonl')
parser.add_argument('--mutate_method', type=str, default='random')
parser.add_argument('--output_file_name', type=str, default='test')
parser.add_argument('--num_round', type=int, default=3)
parser.add_argument('--num_generate', type=int, default=1)
parser.add_argument('--save_seed', type=int , default=1)
parser.add_argument('--recover', type=int, default=0)
parser.add_argument('--calc_analyst', type=int, default=0)
parser.add_argument('--calc_final_result', type=int, default=1)
parser.add_argument('--save_all_seed', type=int, default=0)
parser.add_argument('--clean_data', type=int, default=1)
parser.add_argument('--set_threshold_analyst', type=int, default=1)
parser.add_argument('--calc_relative_reward', type=int, default=1)
parser.add_argument('--clean_mutate_method', type=int, default=1)
parser.add_argument('--MBPP_test_case_num', type=int, default=1)
parser.add_argument('--max_visit_num', type=int, default=1e5)

parser.add_argument('--recover_path', type=str, default='')

parser.add_argument('--parallel', type=int, default=0)
parser.add_argument('--split_input', type=int, default=0)
parser.add_argument('--mutate_level', type=str, default='whole')
parser.add_argument('--llm_critic', type=str, default='none')
parser.add_argument('--with_reference', type=int, default=1)

parser.add_argument('--only_consider_passed_cases', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--workspace', type=str, default='workspace_fuzz')

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

parser.add_argument('--add_monitor', type=int, default=0)
parser.add_argument('--repair_plan', type=int, default=0)
parser.add_argument('--repair_code', type=int, default=0)
parser.add_argument('--run_multi_gen', type=int, default=0)
parser.add_argument('--repair_prompt_num', type=int, default=0)

args = parser.parse_args()
class PromptNode:
    def __init__(self,
                 solution,
                 score=0,
                 passes=0,
                 visited_num=0,
                 reward_score=0,
                 plan_score=[],
                 finish=False,
                 parent: 'PromptNode' = None):

        self.solution = solution


        self.visited_num = visited_num
        self.score=score
        self.passes=passes
        self.plan_score = []
        self.reward_score=reward_score
        self.finish = finish
        

        self.parent: 'PromptNode' = parent
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    # @property
    # def num_jailbreak(self):
    #     return sum(self.results)

    # @property
    # def num_reject(self):
    #     return len(self.results) - sum(self.results)

    # @property
    # def num_query(self):
    #     return len(self.results)

class MCTSExploreSelectPolicy:
    def __init__(self,  initial_seed_len=0,ratio=0.5, alpha=0.1, beta=0.2):

        self.step = 0
        self.mctc_select_path: 'list[PromptNode]' = []
        self.last_choice_index = None
        self.rewards = []
        self.initial_seed_len= initial_seed_len
        self.ratio = ratio  # balance between exploration and exploitation
        self.alpha = alpha  # penalty for level
        self.beta = beta   # minimal reward after penalty

    def select(self,prompt_nodes) -> PromptNode:
        self.step += 1
        if len(prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(prompt_nodes) - len(self.rewards))])
            


        initial_prompts_nodes = prompt_nodes[:self.initial_seed_len]
        # print('printing finish!')
        # print(len(prompt_nodes))
        # print([i.finish for i in prompt_nodes])
        # print(len(initial_prompts_nodes))
        # print([i.finish for i in initial_prompts_nodes])


        initial_prompts_nodes=[prompt_node for prompt_node in initial_prompts_nodes if not prompt_node.finish]


        # print(len(initial_prompts_nodes))


        # # 删除已经结束的节点
        def have_end(prompt_node,initial_prompts_nodes):
            if prompt_node.finish:
                return True
            if len(prompt_node.child)==0:
                return False
            for child in prompt_node.child:
                if have_end(child,initial_prompts_nodes):
                    return True
            return False
        final_initial_prompts_nodes=[]
        for prompt_node in initial_prompts_nodes:
            if have_end(prompt_node,initial_prompts_nodes):
                continue
            if prompt_node.visited_num > args.max_visit_num:
                continue
            final_initial_prompts_nodes.append(prompt_node)
        
        initial_prompts_nodes = final_initial_prompts_nodes
        # print('after select')
        # print(len(initial_prompts_nodes))



        self.mctc_select_path.clear()
        # 第一步一定是在初始的种子里选
         
        cur = max(
            initial_prompts_nodes,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)

        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur.child=[prompt_node for prompt_node in cur.child if not prompt_node.finish]
            cur = max(
                cur.child,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)

        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        print('path & finish')
        print([i.finish for i in self.mctc_select_path])
        return cur

    def update(self, prompt_nodes: 'list[PromptNode]',all_prompt_nodes):
        # succ_num = sum([prompt_node.num_jailbreak
        #                 for prompt_node in prompt_nodes])

        # 新的得分
        reward = sum([prompt_node.reward_score
                        for prompt_node in prompt_nodes])
        
        # print('last_choice_index:'+ str(self.last_choice_index))
        # print('reward: '+str(reward))
        # print(self.mctc_select_path)
        last_choice_node = all_prompt_nodes[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            # 这里reward的计算其实是： jailbreak的概率 * （1-深度*0.1）

            # 如果子节点终止，则所有路径上的节点终止，即该初始种子完成fuzzing
            if prompt_nodes[0].finish:
                prompt_node.finish=True

            reward = reward / len(prompt_nodes)
            self.rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node.level))
        

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
    
    original_node_output_path=args.output_path+'_original_node.jsonl'
    original_prompt_nodes2save=prompt_nodes[:initial_seed_num]
    print('-'*30)
    print('saving original node into: '+original_node_output_path)
    with open(original_node_output_path, 'w+') as f:
        for prompt_node in original_prompt_nodes2save:
            result={
                'visited_num': prompt_node.visited_num,
                'score':prompt_node.score,
                'passes':prompt_node.passes,
                'reward_score':prompt_node.reward_score,
                'finish':prompt_node.finish,
                'level':prompt_node.level,
                'index':prompt_node.index,
                'parent':prompt_node.parent.index if prompt_node.parent else None,
                'child':[child.index for child in prompt_node.child],
                'solution':prompt_node.solution
            }
            f.write(json.dumps(result) + '\n')
            f.flush()
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
                'parent':prompt_node.parent.index if prompt_node.parent else None,
                'child':[child.index for child in prompt_node.child],
                'solution':prompt_node.solution
            }
            f.write(json.dumps(result) + '\n')
            f.flush()
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
    args=None
    ):
    """Run the startup logic. Can be called from CLI or other Python scripts."""
    # return 
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
    if config.agentops_api_key != "":
        agentops.init(config.agentops_api_key, tags=["software_company"])
    print('in generating repo')
    config.set_args(args=args)
    config.update_via_cli(project_path, project_name, inc, reqa_file, max_auto_summarize_code)
    ctx = Context(config=config,args=args)
    ctx.set_args(args)

    if not recover_path:
        # 建立公司，并招募员工
        company = Team(context=ctx)
        # 先找三个员工
        company.hire(
            [
                #再role的初始化函数里就做了llm生成
                ProductManager(args=args),
                Architect(args=args),
                ProjectManager(args=args),
            ]
        )

        if implement or code_review:
            company.hire([Engineer(args=args,n_borg=5, use_code_review=code_review)])

        if run_tests:
            company.hire([QaEngineer(args=args)])
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
    asyncio.run(company.run(args=args,n_round=n_round))

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
    args = None,
    ):
    """Run a startup. Be a boss."""

    if idea is None:
        typer.echo("Missing argument 'IDEA'. Run 'metagpt --help' for more information.")
        raise typer.Exit()
    # print(idea)
    # print('coming to generating repo')
    # print(args)
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
        args=args,
    )


def extract_code_from_repo(file_path):
    files=os.listdir(file_path)
    num_py_files = len(files)
    if num_py_files==0:
        return ''
    # print(files)
    file_name = files[0]
    if 'main' in file_name and num_py_files>1:
        file_name = files[1]
    # print(file_name)
    sourse=''
    code=''
    with open(file_path+ '/'+file_name,'r') as f:
        code = f.read()
    return code
def extract_plan_from_repo(file_path):
    prd_path = file_path +'/docs/prd'
    system_design_path = file_path +'/docs/system_design'
    plan = ''
    RequirementAnalysis,RequirementPool,ImplementationApproach = '','',''
    if not os.path.exists(prd_path) or not os.listdir(prd_path):
        RequirementAnalysis,RequirementPool='',''
    else:
        path = prd_path + '/'+os.listdir(prd_path)[0]
        with open(path,'r') as f:
            try:
                prd=json.load(f)
                RequirementAnalysis = prd['Requirement Analysis']
                RequirementPool = prd['Requirement Pool']
            except:
                pass
    if not os.path.exists(system_design_path) or not os.listdir(system_design_path):
        ImplementationApproach=''
    else:
        path = system_design_path + '/'+os.listdir(system_design_path)[0]
        with open(path,'r') as f:
            try:
                system_design=json.load(f)
                ImplementationApproach = system_design['Implementation approach']
            except:
                pass 
    return RequirementAnalysis,RequirementPool,ImplementationApproach
def delete_repo(file_path):
    import shutil
    shutil.rmtree(file_path) 

def format_plan(RA,RP,IA):
    plan = ''
    if RA:
        plan+=f'requirement analysis:\n{RA}\n'
    if RP:
        plan+='requirement pool:\n'
        for req in RP:
            plan+='- '+req[1]+'\n'
    plan+=IA+'\n'
    return plan
    


if __name__ == '__main__':



    coding_prompt=''
    
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
    # his = format_history(loaded_dataset[0])
    # print(his)
    time.sleep(100)

    # loaded_dataset[:3]

    print(len(loaded_dataset))
    # prompt_nodes=[]
    # passAt10s=[]


    # initial_seed = loaded_dataset
    
    # for i in range(len(initial_seed)):
    #     if initial_seed[i]['pass']:
    #         prompt_nodes.append(PromptNode(initial_seed[i],initial_seed[i]['pass'],initial_seed[i]['pass_num']))
    # print(len(prompt_nodes))
    # initial_seed_num=len(prompt_nodes)
    # for i, prompt_node in enumerate(prompt_nodes):
    #     prompt_node.index = i

    prompt_nodes=[]
    passAt10s=[]
    initial_seed_num=-1
    if args.recover<=0:
        
        

        initial_seed = loaded_dataset

        

        


        for i in range(len(loaded_dataset)):
            # print(initial_seed[i])
            if 'passed' in initial_seed[i].keys():
                if initial_seed[i]['passed']:
                    prompt_nodes.append(PromptNode(initial_seed[i],initial_seed[i]['passed'],initial_seed[i]['pass_num']))
            else:
                if initial_seed[i]['pass']:
                    prompt_nodes.append(PromptNode(initial_seed[i],initial_seed[i]['pass'],initial_seed[i]['pass_num']))
        initial_seed_num = len(prompt_nodes)

        
    else:
        print(f'recoveing from round {args.recover}')
        print(f'loadiing seed from {args.recover_path}')
        def recover_from_jsonl(raws):
            prompt_nodes = []
            for data in raws:
                # "visited_num": 1, "score": true, "passes": 9, "reward_score": 0, "finish": false, "level": 0, "index": 0, "parent": null, "child": [339], "solution":
                parent=None
                if data['parent']:
                    if data['parent']>len(prompt_nodes)-1:
                        print(data['parent'])
                        print(len(prompt_nodes))
                        print(data)
                    parent = prompt_nodes[data['parent']]    
                prompt_nodes.append(PromptNode(
                    visited_num = data['visited_num'],
                    score = data['score'],
                    passes = data['passes'],
                    reward_score = data['reward_score'],
                    finish = data['finish'],
                    solution = data['solution'],
                    parent= parent
                ))
            return prompt_nodes
        raw_datas = []
        with open(args.recover_path+'_original_node.jsonl','r' ) as f:
            lines = [json.loads(line) for line in f]
            for i,line in enumerate(lines):
                if True:
                    raw_datas.append(line)
        print('len of raw_datas of original seed')
        print(len(raw_datas))
        initial_seed_num = len(raw_datas)
        with open(args.recover_path+f'_node_{args.recover-1}.jsonl','r' ) as f:
            lines = [json.loads(line) for line in f]
            raw_datas+=lines
        print('len of raw_datas of all seed')
        print(len(raw_datas))
        
        prompt_nodes = recover_from_jsonl(raw_datas)
        record(args,prompt_nodes,initial_seed_num)

    for i, prompt_node in enumerate(prompt_nodes):
        prompt_node.index = i
    

    num_seed=len(prompt_nodes)
    # print(initial_seed_num)
    print(f'len of loaded seeds: {initial_seed_num}')

    select_policy = MCTSExploreSelectPolicy(len(prompt_nodes))

    from sentence_transformers import SentenceTransformer, util

    # 加载预训练的Sentence-BERT模型
    if args.calc_analyst==1:
        semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cuda:{}'.format(0))


    num_seed =len(prompt_nodes)
    # text, code, task_id, test_list, entry_point

    fail_list = []
    output_path = args.output_path + 'HumanEval_ET.jsonl'
    output_codes=[]
    threshold=args.num_round
    for idx in range(args.recover,args.recover+threshold):
        print('----'*10+'round: '+str(idx)+'---'*10)

        print('-'*10+'selecting seed'+'-'*10)
        seed = select_policy.select(prompt_nodes)
        print('current seed index:' + str(seed.index))

        # print(seed.score)
        # print(seed.reward_score)


        print('-'*10+'mutating'+'-'*10)
        if 'prompt' not in seed.solution.keys():
            seed.solution['prompt']=seed.solution['description']
        if args.split_input==1:
            mutated_seed,cur_mutate_method = mutate_one_nl(seed,args,args.mutate_method)
        else:
            mutated_seed,cur_mutate_method = mutate_one(seed,args,args.mutate_method)
        # print(seed.solution['prompt'])
        # print(mutated_results)
        print('-'*10+'evaluating mutated seed'+'-'*10)


        
        # if not os.path.exists(code_output_path):
        #     os.mkdir(code_output_path)

        # text code task_id entry_point

        score, passes=-1,-1
        task=mutated_seed.solution
        code_output_path=args.output_path+'code'+'_round_'+str(idx)+'.jsonl'
        with open(code_output_path, 'w+') as f:
            # args.num_generate=10
        
            codes = []
            plans=[]
            # intent=task['prompt']
            # method_name = task['entry_point']

            before_func=''
            if 'prompt' not in task.keys():
                if 'description' in task.keys():
                    task['prompt'] = task['description']
                elif 'text' in task.keys():
                    task['prompt'] = task['text']
                else:
                    raise NotImplementedError
            if 'human' in args.dataset or 'mbpp' in args.dataset:
                method_name = task['entry_point']
            elif 'codecontest' in args.dataset:
                method_name = f'codecontest_{idx}'
            else:
                raise NotImplementedError
            method_name = f'{args.dataset}_{idx}'
            intent = task['prompt']
            before_func=''
            repair_prompt = []
            mutation_method = ['expand_one2two','condense_two2one','rephrase_one']
            if args.run_multi_gen==1:
                for i in range(args.repair_prompt_num):
                    new_prompt= get_more_prompt_test(intent,args,mutation_method[i])
                    print('multi-gen-prompt:')
                    print(new_prompt)
                    repair_prompt.append(new_prompt)
            task['repair_prompt'] = [intent]+repair_prompt

            def form_prompt_mbpp(input_propmt,task):
                test_cases = task['test_list'][:args.MBPP_test_case_num]
                # TEST_PROMPT = '\nexample:\n'
                split_index = input_propmt.find('function ') + len('function ')
                method_name = task['entry_point']
                input_propmt = input_propmt[:split_index] + method_name + ' ' + input_propmt[split_index:]
                TEST_PROMPT = '\n'
                input_propmt =  input_propmt+TEST_PROMPT
                for test in test_cases:
                    input_propmt+=test[7:]+'\n'
                return input_propmt
            def form_prompt_codecontest(intent):
                codecontest_coding_prompt ='\n-------\nImportant Note: You must follow the input output format.  The code will be tested against multiple test cases and all the test cases must be passed.'
                if '\nExample\n' in intent:
                    i = intent.find('\nExample\n')
                    prompt = intent[:i]+codecontest_coding_prompt+intent[i:]
                else:
                    prompt = intent+codecontest_coding_prompt
                if 'deepseek' in args.model:
                    prompt +=  '''\nWrite a main() function and use input() function to read input from stdin'''
                    p="\nPlease use 'if __name__ == '__main__':' and 'input()' function to read input\n"
                return prompt
            
            for i in range(len(task['repair_prompt'])):
                if 'mbpp' in args.dataset:
                    task['repair_prompt'][i] = form_prompt_mbpp(task['repair_prompt'][i],task)
                    # print(task['repair_prompt'][0])
                if 'codecontest' in args.dataset:
                    task['repair_prompt'][i] = form_prompt_codecontest(task['repair_prompt'][i]) 
            
            if 'human' in args.dataset:
                before_func,code_in_prompt = prompt_split_humaneval(intent,method_name)
                code_in_prompt = code_in_prompt.split('\n')[0]

            # new_loop = asyncio.new_event_loop()
            # asyncio.set_event_loop(new_loop)
            # repo=startup(idea=coding_prompt+intent,project_name=method_name)
            # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ method_name+'/'+method_name
            # code = extract_code_from_repo(file_path,intent)
            # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ method_name
            # delete_repo(file_path)
            # print(code)
            # output_codes.append(code)

            def generate(method_name,task):
                try:
                    # 进行分工流程在这里，输入了prompt为intent
                    futures=[]
                    split_para=1
                    if args.parallel==1:
                        split_para=1
                        for __ in range(split_para):
                            with ProcessPoolExecutor() as executor:
                                for cnt in range(args.num_generate):
                                    print('in generating...')
                                    ii = (cnt//len(task['repair_prompt']))%len(task['repair_prompt'])
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    new_method_name = method_name+"_"+str(cnt)
                                    # repo=startup(idea=coding_prompt+intent,project_name=method_name)
                                    # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name+'/'+new_method_name
                                    # code = extract_code_from_repo(file_path,intent)
                                    # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name
                                    # delete_repo(file_path)
                                    # future= executor.submit(session.run_session,need_second_round,finally_pass)
                                    # print(f'idea: {coding_prompt+intent}')
                                    # print('in generating repo...')
                                    idea = coding_prompt+task['repair_prompt'][ii]
                                    # print(f'idea: {idea}')
                                    args_dict = vars(args)
                                    future= executor.submit(startup,idea=idea,project_name=new_method_name,args=args_dict)
                                
                                    futures.append(future)
                        results=[]
                        for cnt, future in enumerate(as_completed(futures)):
                            # print('-'*80)
                            # # print(len(future))
                            # print(generate_ids)
                            # print(cnt)
                            # print(generate_ids[cnt])
                            results.append(future.result())

                        
                            # session_historys.append(session_history)
                    else:
                        pass
                except Exception as e:
                    print(str(e))
            
            def read_code_plans(method_name):
                for cnt in range(args.num_generate):
                    # print(future.result())
                    new_method_name = method_name+"_"+str(cnt)
                    code_file_path = '/home/zlyuaj/muti-agent/MetaGPT/{}/'.format(args.workspace)+ new_method_name+'/'+new_method_name
                    if not os.path.exists(code_file_path):
                        print(f'code file path not exist! {code_file_path}')
                        continue
                    code = extract_code_from_repo(code_file_path)
                    plan_file_path = '/home/zlyuaj/muti-agent/MetaGPT/{}/'.format(args.workspace)+ new_method_name
                    if not os.path.exists(plan_file_path):
                        continue
                    RA,RP,IA = extract_plan_from_repo(plan_file_path)
                    # print(RequirementAnalysis,RequirementPool,ImplementationApproach)
                    delete_repo(plan_file_path)
                    
                    if not code or (not RA and not RP and not IA):
                        continue
                    plan = format_plan(RA,RP,IA)
                    code = 'from typing import List\n'+code
                    import re
                    def remove_input_content(code):
                    # 使用正则表达式替换 input() 中的内容
                        code = re.sub(r'input\([^)]*\)', 'input()', code)
                        return code
                    code = remove_input_content(code)
                    # code, session_history=future.result()
                    # print('#'*30)
                    # print(code)
                    codes.append(code)
                    plans.append(plan)
                        # output_codes.append(code)
                        # print(code)
                return codes,plans
            

            max_try=3
            for _ in range(max_try):
                generate(method_name=method_name,task=task)
                codes,plans = read_code_plans(method_name)
                # print('-'*100)
                # print(codes)
                # print('-'*100)
                # print(plans)
                if len(codes)>0:
                    break
            # generate_ids=[i for i in range(args.num_generate)]
            # loop_count =0
            # while len(generate_ids)>0 and loop_count<3:
            #     codes,all_RequirementAnalysis,all_RequirementPool,all_ImplementationApproach,generate_ids = generate(method_name,intent,generate_ids,all_RequirementAnalysis,all_RequirementPool,all_ImplementationApproach)
            #     loop_count +=1
            #     print(generate_ids)
            # if loop_count==3:
            #     print('fail to generate 10 code output!')
            #     print('round number: {}'.format(idx))
            #     print('task_id: {}'.format(task['task_id']))
            # codes.append(code)
            run_eval=True
            if not codes:
                print(f'no answer for question {idx}')
                codes = ['']*10
                plans=['']*10
                run_eval = False
            else:
                while len(codes)<args.num_generate:
                    ran = random.randint(0,len(codes)-1)
                    codes.append(codes[ran])
                    plans.append(plans[ran])
            task['completions']=codes
            task['plans'] = plans
            task['mutate_method'] = cur_mutate_method


            if run_eval:
                print('evaluating ...')
                if 'human' in args.dataset:
                    score, passes,passAt1, passAt10= evaluate_one(task,args.num_generate)
                elif 'mbpp' in args.dataset:
                    score, passes,passAt1, passAt10= evaluate_one_MBPP(task,args.num_generate)
                elif 'codecontest' in args.dataset:
                    score, passes,passAt1, passAt10= evaluate_one_codecontest(task,args.num_generate)
            else:
                score, passes,passAt1, passAt10 = False,0,False,False
            # mutated_seed.score=score
            
            # print('evaluating ...')
            # score, passes,passAt1, passAt10= evaluate_one(task,args.num_generate)
            mutated_seed.score=passAt10
            mutated_seed.passes=passes

            
            print(passAt10)
            task['pass'] = passAt10
            task['pass_num'] = passes
            task['round']=idx
            task['parent_index'] = seed.index
            # entry_point = find_method_name(code)
            f.write(json.dumps(task) + '\n')
            f.flush()

        mutated_seed.solution=task

        print('-'*10+'updating'+'-'*10)

        # 如果pass@10不为真，那么说明fuzzing已经结束，将其reawrd设置为-1
        if mutated_seed.score==False:
            mutated_seed.reward_score = -1e4
            mutated_seed.finish = True
            seed.finish = True
            mutated_seed.index = len(prompt_nodes)
            prompt_nodes.append(mutated_seed)
            print('seed '+ str(seed.index) + ' finish fuzzing!')
            print('seed_index: '+str(seed.index))
            print('mutated_seed_index: '+str(mutated_seed.index))
            # print('-'*30)
            num_seed-=1
            print('current seed length: '+ str(num_seed))
            

            
        else:

            if args.save_seed==1:
                if args.save_all_seed==1:
                    print('save all seed without reward!')
                    mutated_seed.index = len(prompt_nodes)
                    prompt_nodes.append(mutated_seed)
                    print('add mutated seed into prompt node list')
                else:
                # 加入判断条件，即pass10的通过变少
                    # print(' in calc analyst')
                    analyst_reward,final_output_reward=0,0
                    # 计算analyst的reward
                    if args.calc_analyst==1:
                        seed_plans=seed.solution['plans']
                        mutated_seed_plans=mutated_seed.solution['plans']

                        seed_plans=semantic_model.encode(seed_plans, convert_to_tensor=True)
                        mutated_seed_plans=semantic_model.encode(mutated_seed_plans, convert_to_tensor=True)
                        seed_plans = F.normalize(seed_plans, dim = 1)
                        mutated_seed_plans = F.normalize(mutated_seed_plans, dim = 1)
                        similarity = torch.einsum("ab,cb->ac",seed_plans,mutated_seed_plans)
                        similarity=torch.mean(similarity)
                        similarity = float(similarity)
                        # for sc in semantic_scores:
                    #     print(sc)
                        analyst_score = 1- similarity
                        # print('---'*10)
                        # print('analyst score')
                        # print(analyst_score)
                        # print('---'*10)
                        if args.set_threshold_analyst==1:
                            if analyst_score > 0.1:
                                analyst_reward = analyst_score
                            else:
                                analyst_reward = 0




                        
                        




                            
                    # 计算最终输出的reward
                    if args.calc_final_result==1:  
                        if args.calc_relative_reward==1:
                            final_output_reward=max(0,(seed.passes-mutated_seed.passes) / 10)
                        else:
                            final_output_reward = 1-mutated_seed.passes / 10



                    reward = args.beta * (args.alpha*analyst_reward + final_output_reward) 

                    print('analyst reward: {}'.format(analyst_reward))
                    print('final_output reward: {}'.format(final_output_reward))
                    print('total reward: {}'.format(reward))

                    
                    if  reward>0:
                        mutated_seed.reward_score = reward
                        mutated_seed.index = len(prompt_nodes)
                        prompt_nodes.append(mutated_seed)
                        
                        print('add mutated seed into prompt node list')
                        print('seed_index: '+str(seed.index))
                        print('mutated_seed_index: '+str(mutated_seed.index))
                        # print('-'*30)
                    
            else:
                print('do not save any seed')
               
            print('reward = {}'.format(mutated_seed.reward_score))

        select_policy.update([mutated_seed],prompt_nodes)


        

        output_file = args.output_path + 'log.txt'
        with open(output_file,'a') as f:
            f.write('round:'+str(idx)+'\n')
            f.write('current node number: '+str(len(prompt_nodes))+'\n')
            f.write('old score: '+str(seed.score) + ' ,new score: '+str(mutated_seed.score) + '\n')
            f.write('seed prompt:\n ' + seed.solution['prompt']+'\n')
            f.write('mutated prompt:\n' + mutated_seed.solution['prompt']+'\n\n\n\n')
        

        print('saving......')
        if (idx-1) % 10==0:
            cur_passAt10=record(args,prompt_nodes,initial_seed_num)
            passAt10s.append(cur_passAt10)


        if (idx-1)%10==0:
            save_node(args,prompt_nodes,initial_seed_num,idx)
    print('fuzzing finished!')
    print('total prompt nodes number:' + str(len(prompt_nodes)))

    save_node(args,prompt_nodes,initial_seed_num,1000)
    node_output_path=args.output_path+'final_node.jsonl'
    with open(node_output_path, 'w+') as f:
        for prompt_node in prompt_nodes:
            result={
                'task_id':prompt_node.solution['task_id'],
                'prompt':prompt_node.solution['prompt'],
                'pass@10':prompt_node.score,
                'passes':prompt_node.passes,

            }
            f.write(json.dumps(result) + '\n')
            f.flush()

    cur_passAt10=record(args,prompt_nodes,initial_seed_num)
    passAt10s.append(cur_passAt10)
    final_reuslt_output_path=args.output_path+'final_result.jsonl'
    with open(final_reuslt_output_path, 'a') as f:
        f.write(str(passAt10s)+'\n')
        f.flush()



