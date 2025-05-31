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
from session import Session
from datasets import load_dataset, load_from_disk
from utils import prompt_split_humaneval, find_method_name, code_split, build_test_method
from evaluate_result import evaluate_all,evaluate_one
from main_mutate import mutate_one,mutate_one_nl,get_more_prompt_test
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='humaneval')
parser.add_argument('--lang', type=str, default='python')
parser.add_argument('--output_path', type=str, default='output.jsonl')
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
parser.add_argument('--clean_data', type=int, default=0)
parser.add_argument('--set_threshold_analyst', type=int, default=1)
parser.add_argument('--calc_relative_reward', type=int, default=1)
parser.add_argument('--calc_llm_absolute', type=int, default=0)
parser.add_argument('--single_llm', type=int, default=0)
parser.add_argument('--clean_mutate_method', type=int, default=0)
parser.add_argument('--add_monitor', type=int, default=0)
parser.add_argument('--repair_plan', type=int, default=0)
parser.add_argument('--repair_code', type=int, default=0)
parser.add_argument('--run_multi_gen', type=int, default=0)
parser.add_argument('--repair_prompt_num', type=int, default=0)
parser.add_argument('--max_visit_num', type=int, default=1e5)

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


import openai
from openai import OpenAI
from openai import AzureOpenAI
def call_chatgpt(prompt, model='gpt-35-turbo', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, echo=False, majority_at=None):
    # print('$$$'*200)
    # print('in call gpt')
    # print(model)
    # client = OpenAI()
    client = AzureOpenAI(
    azure_endpoint = "https://hkust.azure-api.net", 
    api_key="b234b6eb250e445d8151e8e5710dadde",  
    api_version="2024-02-01"
    )

    # client = AzureOpenAI(
    # azure_endpoint = "https://hkust.azure-api.net", 
    # api_key="b8927c969e8147ea8404003613bbddb6",  
    # api_version="2024-02-01"
    # )
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10

    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        
        try:
            # print('***'*30)
            # print(prompt)
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            # print(client.api_key)
            # print(client.base_url)
            # print(max_tokens,temperature,top_p,requested_completions)
            response = client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=requested_completions
            )
            while not response:
                client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions
                    )
            completions.extend([choice.message.content for choice in response.choices])
            # print(completions[0])
            # print('*'*30)
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.RateLimitError as e:
            time.sleep(min(i**2, 60))
    raise RuntimeError('Failed to call GPT API')



class PromptNode:
    def __init__(self,
                 solution,
                 score=0,
                 passes=0,
                 plan_score=[],
                 parent: 'PromptNode' = None):

        self.solution = solution


        self.visited_num = 0
        self.score=score
        self.passes=passes
        self.plan_score = []
        self.reward_score=0
        self.finish = False
        

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
        if not final_initial_prompts_nodes:
            return None
        
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
    max_score=1
    max_passes=1000
    worst_scores=[max_score]*initial_seed_num
    worst_passes=[max_passes]*initial_seed_num
    # for prompt_node in prompt_nodes:
    #     task_id=int(prompt_node.solution['task_id'].split('/')[-1])
    #     worst_scores[task_id]=min(worst_scores[task_id],prompt_node.score)
    #     worst_passes[task_id]=min(worst_passes[task_id],prompt_node.passes)
    
    # print('worst_scores')
    # print(worst_scores)
    total_passAt10=[prompt_node.finish for prompt_node in prompt_nodes[:initial_seed_num]].count(False)
    print('total pass@10:' + str(total_passAt10))
    # print('worst_passes')
    # print(worst_passes)
    worst_avg_score=round(sum(worst_scores)/len(worst_scores),4)
    # print(worst_avg_score)
    total_passes=sum(worst_passes)
    # print(sum(worst_passes))

    initial_avg_score=round(sum(initial_score)/len(worst_scores),4)
    initial_total_passes=sum(initial_passes)

    score_difference=worst_avg_score-initial_avg_score
    passes_difference=total_passes-initial_total_passes

    final_reuslt_output_path=args.output_path+'_final_result.jsonl'
    print('-'*30)
    print('saving result into: '+final_reuslt_output_path)
    with open(final_reuslt_output_path, 'a') as f:
        # result={
        #     'worst_avg_scores':worst_avg_score,
        #     'total_pass@10':total_passAt10,
        #     'final_total_passes':total_passes,
        #     'initial_avg_score':initial_avg_score,
        #     'initial_total_passes':initial_total_passes,
        #     'score_difference':score_difference,
        #     'passes_difference':passes_difference,
        #     'worst_scores':worst_scores,
        #     'worst_passes':worst_passes
            

        # }
        # f.write(json.dumps(result) + '\n')
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
                'index':prompt_node.index,
                'parent':prompt_node.parent.index,
                'child':[child.index for child in prompt_node.child],
                'score':prompt_node.score,
                'passes':prompt_node.passes,
                'mutate_method':prompt_node.solution['mutate_method'],
                'prompt':prompt_node.solution['prompt'],
                'completions':prompt_node.solution['completions'],
                'plan':[history['plan'] if 'plan' in history.keys() else '' for history in prompt_node.solution['session_historys'] ],
                'task_id':prompt_node.solution['task_id']
            }
            f.write(json.dumps(result) + '\n')
            f.flush()
if __name__ == '__main__':
    from roles.rule_descriptions_actc import TEAM, ANALYST, PYTHON_DEVELOPER, TESTER

    
    
    initial_output_path=args.output_path
    args.output_path=initial_output_path+'results-'+args.output_file_name+'/'
    x=2
    while os.path.exists(args.output_path):
        args.output_path=initial_output_path+'results-'+args.output_file_name+'_'+str(x)+'/'
        x+=1
    os.mkdir(args.output_path)
    print(args.output_path)
    print(args)
 
    from datasets import load_dataset
    # ds = load_dataset("dz1/CodeScore-MBPP-ET")
# Failed to call GPT API                                                                                   
# analyze fail


    # load dataset
    INPUTPATH=args.input_path
    loaded_dataset=[]
    with open(INPUTPATH, 'r') as f:
        # 导入输出
       loaded_dataset = [json.loads(line) for line in f]
    

    prompt_nodes=[]
    passAt10s=[]

    initial_seed = loaded_dataset
    initial_seed_num=len(loaded_dataset)
    # if args.recover==0 :
    # if 'human'  in args.dataset and '35' in args.model:
    #     initial_score=[1.0, 1.0, 0.165, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6929, 0.0664, 0.0315, 0.7325, 0.9915, 1.233, 1.0, 1.0, 1.0, 1.0, 0.8248, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9587, 0.3276, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.669, 1.0, 0.9921, 0.5084, 1.0, 0.1098, 1.0, 1.0, 0.1143, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0992, 1.0, 1.0, 0.9906, 0.7278, 0.9849, 0.9765, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9452, 0.8012, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8964, 0.8815, 0.2639, 1.0, 1.0, 0.97, 0.4123, 1.0, 0.0, 0.6038, 1.0, 0.9826, 0.7172, 0.9952, 0.7, 0.6517, 0.2897, 1.0, 0.2692, 0.9925, 0.8853, 0.0074, 0.9074, 0.96, 0.9054, 0.4987, 0.7928, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9041, 0.2708, 0.9189, 0.4534, 0.9516, 1.0, 0.8116, 1.0, 0.0031, 1.0, 1.0, 0.7069, 0.9359, 0.2113, 1.0, 0.9121, 1.0, 0.8671, 0.2547, 0.7, 0.9496, 0.9182, 0.0721, 0.0, 0.9118, 0.7, 1.0, 0.9788, 0.8396, 1.0, 0.5839, 0.0977, 1.0, 0.8016, 1.0, 0.159, 1.0, 0.9977, 0.3962, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9463, 1.0, 1.0, 0.854, 0.9798, 1.0, 1.0, 1.0, 0.0894, 0.7062, 1.0, 1.0, 0.0917]
    #     initial_passes=[105.0, 69.0, 17.0, 91.0, 105.0, 110.0, 99.0, 106.0, 94.0, 59.6, 7.7, 4.0, 67.9, 116.0, 127.0, 48.0, 128.3, 129.0, 132.0, 99.8, 130.0, 130.0, 87.0, 94.0, 33.0, 72.8, 22.0, 126.0, 87.0, 112.0, 96.0, 70.0, 0.0, 106.8, 126.0, 125.0, 58.2, 127.0, 125.0, 6.5, 126.0, 10.2, 85.0, 126.0, 15.2, 73.0, 74.0, 111.0, 131.0, 120.0, 12.5, 133.0, 126.0, 105.0, 81.0, 32.4, 80.9, 113.0, 100.0, 105.0, 102.0, 79.0, 99.0, 30.0, 108.7, 76.5, 113.0, 115.0, 99.0, 141.0, 107.0, 114.0, 103.0, 102.0, 107.0, 51.9, 86.2, 19.0, 105.9, 30.0, 105.6, 43.7, 136.0, 0.1, 73.7, 95.0, 113.0, 54.6, 83.0, 56.5, 57.1, 31.0, 131.0, 35.0, 133.0, 110.4, 1.0, 98.0, 52.8, 129.8, 39.5, 88.0, 116.0, 119.0, 131.0, 106.0, 30.0, 111.6, 26.0, 106.0, 72.2, 118.0, 135.0, 116.1, 120.0, 0.1, 113.0, 133.0, 89.8, 87.0, 30.0, 133.0, 124.0, 73.0, 133.7, 34.0, 77.0, 127.0, 95.6, 8.2, 0.0, 95.0, 84.7, 120.0, 134.1, 89.0, 124.0, 65.9, 29.7, 16.0, 121.3, 126.1, 8.0, 133.0, 129.7, 42.0, 102.0, 98.0, 102.0, 133.0, 137.0, 85.8, 132.0, 135.0, 112.2, 77.8, 132.0, 134.0, 137.0, 28.2, 86.9, 128.0, 131.0, 11.1]
    #     initial_score = [True, True, False, True, True, True, True, True, True, True, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, False, False, True, True, True, True, False, True, True, True, True, True, False, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, False, True, False, True, True, False, True, True, True, False, False, True, False, False, True, False, False, True, True, False, False, True, True, True, True, True, True, False, True, True, False, True, True, True, False, True, True, False, True, False, True, True, True, True, False, False, True, True, False, False, True, False, True, True, False, True, False, False, True, False, True, False, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, False]
    #     initial_passes = [10, 10, 0, 10, 10, 10, 10, 10, 10, 6, 0, 0, 0, 0, 0, 10, 9, 10, 10, 8, 10, 10, 10, 10, 10, 9, 0, 10, 10, 10, 10, 10, 0, 8, 10, 10, 6, 10, 0, 0, 10, 1, 10, 10, 0, 10, 10, 10, 10, 10, 0, 10, 10, 0, 0, 4, 9, 10, 10, 10, 10, 10, 10, 10, 1, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 9, 10, 6, 0, 10, 0, 7, 10, 0, 2, 10, 5, 0, 0, 10, 0, 0, 8, 0, 0, 7, 9, 0, 0, 10, 10, 10, 10, 10, 2, 0, 10, 2, 0, 10, 9, 10, 0, 10, 10, 0, 2, 0, 10, 10, 10, 9, 0, 0, 10, 2, 0, 0, 3, 0, 10, 9, 0, 10, 0, 0, 10, 0, 9, 0, 10, 9, 0, 10, 10, 10, 10, 10, 0, 10, 10, 4, 4, 10, 10, 10, 1, 4, 10, 10, 0]
    #     should_delete=[0, 2, 3, 4, 5, 9, 11, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 33, 34, 35, 36, 37, 39, 44, 45, 47, 58, 60, 65, 66, 73, 74, 75, 76, 79, 89, 92, 94, 96, 97, 100, 101, 102, 103, 105, 108, 113, 114, 116]
    #     initial_score=initial_score[:initial_seed_num]
    #     initial_passes=initial_passes[:initial_seed_num]
    #     # prompt_nodes = [
    #     #         PromptNode(initial_seed[i],initial_score[i],initial_passes[i]) for i in range(len(initial_seed))
    #     #     ]
    # else:
    initial_score = [0 for i in range(initial_seed_num)]
    initial_passes = [0 for i in range(initial_seed_num)]
    should_delete = [False for i in range(initial_seed_num)]
    prompt_nodes=[]
    print(len(initial_score),len(initial_seed),len(initial_passes))
    need_to_delete = [78,113]
    for i in range(len(initial_score)):
        if 'turbo' in args.model:
            if initial_score[i] and i not in need_to_delete:
                prompt_nodes.append(PromptNode(initial_seed[i],initial_score[i],initial_passes[i]))
        else:
            if initial_seed[i]['pass'] and i not in need_to_delete:
                prompt_nodes.append(PromptNode(initial_seed[i],initial_seed[i]['pass'],initial_seed[i]['pass_num']))

    # print(len(prompt_nodes))


    
    # print()
    if args.clean_data==1:
        prompt_nodes = [prompt_nodes[i] for i in range(len(prompt_nodes)) if i not in should_delete]
    # print('-'*30)
    # print([int(i.solution['task_id'].split('/')[1]) for  i in prompt_nodes] )
    # print('-'*30)

    
    for i, prompt_node in enumerate(prompt_nodes):
        prompt_node.index = i
    initial_seed_num = len(prompt_nodes)
    num_seed=len(prompt_nodes)
    print(initial_seed_num)
    
    invalid_cnt = 0
    
    select_policy = MCTSExploreSelectPolicy(len(prompt_nodes))

    from sentence_transformers import SentenceTransformer, util

    # 加载预训练的Sentence-BERT模型
    if args.calc_analyst==1:
        semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cuda:{}'.format(0))




    fail_list=[]
    threshold=args.num_round
    
    for idx in range(args.recover,args.recover+threshold):
        print('----'*10+'round: '+str(idx)+'---'*10)

        print('-'*10+'selecting seed'+'-'*10)
        seed = select_policy.select(prompt_nodes)
        if not seed:
            break
        print('current seed index:' + str(seed.index))



        print('-'*10+'mutating'+'-'*10)
        # if args.split_input==1:
        mutated_seed,cur_mutate_method = mutate_one_nl(seed,args,args.mutate_method)
        # else:
        #     mutated_seed,cur_mutate_method = mutate_one(seed,args,args.mutate_method)
        # print(seed.solution['prompt'])
        # print(mutated_results)
        print('-'*10+'evaluating mutated seed'+'-'*10)


        task=mutated_seed.solution
        intent=task['prompt']
        method_name = task['entry_point']
        before_func = prompt_split_humaneval(task['prompt'],method_name)
        repair_prompt = []
        mutation_method = ['expand_one2two','condense_two2one','rephrase_one','add_one_sentence']
        if args.run_multi_gen==1:
            for i in range(args.repair_prompt_num):
                new_prompt= get_more_prompt_test(intent,args,mutation_method[i])
                print('multi-gen-prompt:')
                print(new_prompt)
                repair_prompt.append(new_prompt)
        task['repair_prompt'] = [intent]+repair_prompt


        score, passes=-1,-1
        code_output_path=args.output_path+'code'+'_round_'+str(idx)+'.jsonl'
        with open(code_output_path, 'w+') as f:
            codes=[]
            session_historys=[]
            
            need_second_round, finally_pass=0,0
            try:
                # 进行分工流程在这里，输入了prompt为intent
                futures = []
                # if args.parallel == 1:
                with ProcessPoolExecutor() as executor:
                    for cnt in range(args.num_generate):
                        ii = (cnt//len(task['repair_prompt']))%len(task['repair_prompt'])
                        session = Session(TEAM, ANALYST, PYTHON_DEVELOPER, TESTER,requirement=task['repair_prompt'][ii], model=args.model, repair_prompt=bool(args.repair_plan),repair_code=bool(args.repair_code),add_monitor = bool(args.add_monitor), majority=args.majority, 
                                    max_tokens=args.max_tokens, temperature=args.temperature, 
                                    top_p=args.top_p, max_round=args.max_round, before_func=before_func)
                        # future= executor.submit(session.run_session,need_second_round,finally_pass)
                        future= executor.submit(session.run_session,need_second_round,finally_pass)

                        futures.append(future)
                for _, future in enumerate(as_completed(futures)):
                    # print(future.result())
                    code, session_history, need_second_round, finally_pass=future.result()
                    codes.append(code)
                    session_historys.append(session_history)

                    
                # else:
                #     for cnt in range(args.num_generate):
                #         session = Session(TEAM, ANALYST, PYTHON_DEVELOPER, TESTER,requirement=intent, model=args.model, majority=args.majority, 
                #                     max_tokens=args.max_tokens, temperature=args.temperature, 
                #                     top_p=args.top_p, max_round=args.max_round, before_func=before_func)
                        
                #         if args.single_llm==1:
                #             code, session_history, need_second_round, finally_pass = session.run_coder_only()
                #         else:
                #             code, session_history, need_second_round, finally_pass = session.run_session(need_second_round,finally_pass)
                        
                        
                #         codes.append(code)
                #         session_historys.append(session_history)


            except RuntimeError as e:
                print(str(e))
                print("task-%d fail"%(task['task_id']))
                fail_list.append(task['task_id'])
                continue
            

            task['completion']=code
            task['completions']=codes
            task['session_history']=session_history
            task['session_historys']=session_historys
            task['mutate_method'] = cur_mutate_method
            
            # score是所有生成的代码在test case上的平均得分
            # passes是通过所有测试的代码数量（1-10）
            score, passes,passAt1, passAt10= evaluate_one(task,args.num_generate)
            # mutated_seed.score=score
            if score ==-1:
                invalid_cnt+=1
                print('invalid (jailbreak or other issue filter by azure)')
                print('invalid_cnt: '+str(invalid_cnt))
                # 跳过后面保存的环节
                continue

            mutated_seed.score=passAt10
            mutated_seed.passes=passes
            
            if mutated_seed.passes<seed.passes:
                task['save_node'] = True
            else:
                task['save_node'] = False
            task['pass'] = passAt10
            task['parent_index'] = seed.index
            task['round']=idx
            entry_point = find_method_name(code)
                    
            
            
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
                        seed_plans=[]
                        if args.calc_original_plan==1:
                            original_seed = seed
                            # print('testing...')
                            # print(initial_seed_num)
                            while original_seed.index>=initial_seed_num:
                                original_seed = original_seed.parent
                            seed_plans=[history['plan'] for history in original_seed.solution['session_historys']]
                        else:
                            seed_plans=[history['plan'] for history in seed.solution['session_historys']]
                        mutated_seed_plans=[history['plan'] for history in mutated_seed.solution['session_historys']]
                        if args.only_consider_passed_cases==1:
                            seed_plans = [seed_plans[i] for i in range(len(seed_plans)) if seed.solution['pass_results'][i]]
                            mutated_seed_plans = [mutated_seed_plans[i] for i in range(len(mutated_seed_plans)) if mutated_seed.solution['pass_results'][i]]
                        if args.llm_critic == 'none':
                            # print('xxxxxxxxxxxxxxxxxxxxxxxxx')
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
        # print('len(prompt_nodes): '+str(len(prompt_nodes)))
        # print('finish  before fuzzing')
        # print(select_policy.rewards)
        # print([prompt_node.finish for prompt_node in prompt_nodes])
        select_policy.update([mutated_seed],prompt_nodes)
        # print('finish after fuzzing')
        # # print(select_policy.rewards)
        # print([prompt_node.finish for prompt_node in prompt_nodes])

            

        

        # print('len(prompt_nodes): '+str(len(prompt_nodes)))
        # print('reward before fuzzing')
        # # print(select_policy.rewards)
        
        # # select_policy.update([mutated_seed],prompt_nodes)
        # select_policy.rewards[seed.index]=seed.reward_score
        # # print('reward after fuzzing')
        # print([prompt_node.finish for prompt_node in prompt_nodes])
        # print(select_policy.rewards)

        

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


    

        
    

