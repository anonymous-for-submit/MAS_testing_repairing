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
import multiprocessing
from evaluate_result import evaluate_one_MBPP,evaluate_one
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
parser.add_argument('--recover_path', type=str, default='')
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



from code_contests.data.provider import CodeContestDataProvider
from gen.coding_competitor import CodeContestsCompetitor
from gen.utils import evaluate_solution_on_subset
from log import setup_logger
from settings.config_loader import get_settings
from gen.dataset_solver import solve_dataset,solve_dataset_one_problem,solve_dataset_one_problem_parallel
import toml
import datetime

if __name__ == '__main__':

    
    
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



    data_provider = CodeContestDataProvider(dataset_location=args.dataset)
    num_problems = len(data_provider.dataset[args.split_name])
    # print(f'num_problems: {num_problems}')

    # load dataset
    INPUTPATH=args.input_path
    
    comletions_plans_passes=[]
    with open(INPUTPATH, 'r') as f:
        # 导入输出
        comletions_plans_passes = [json.loads(line) for line in f]

    if 'human' in args.dataset and 'nl' not in comletions_plans_passes[0].keys():
        print('loading nl, func, examples from original dataset')
        original_path = '/data/zlyuaj/muti-agent/PairCoder/data/HumanEval_test_case_ET.jsonl'
        with open(original_path, 'r') as f:
        # 导入输出
            original_data = [json.loads(line) for line in f]
            original_data_dict= {data['task_id']:data for data in original_data}
            for i in range(len(comletions_plans_passes)):
                data = original_data_dict[comletions_plans_passes[i]['name']]
                comletions_plans_passes[i]['nl'] = data['nl']
                comletions_plans_passes[i]['func'] = data['func']
                comletions_plans_passes[i]['examples'] = data['examples']


    if 'human' in args.dataset or 'mbpp' in args.dataset:
        loaded_dataset = comletions_plans_passes
    else:
        loaded_dataset = []
        for problem_number in range(num_problems):
            problem_name = data_provider.dataset[args.split_name][int(problem_number)]['name']

            # 从导入的dataset里面找到问题
            problem = data_provider.find_problem(ds=data_provider.dataset, problem_name=problem_name, split_name=args.split_name)
            problem['dataset_name'] = args.dataset
            loaded_dataset.append(problem)



        

        # assert len(comletions_plans_passes) == len(loaded_dataset)

        loaded_dataset = loaded_dataset[:len(comletions_plans_passes)]
        if 'mbpp' in args.dataset and 'Mbpp' not in comletions_plans_passes[0]['name']:
            data_dict = {'Mbpp/'+str(data['name'].split('/')[1]):data for data in comletions_plans_passes}
        else:
            data_dict = {data['name']:data for data in comletions_plans_passes}
        # print(data_dict.keys())
        for i in range(len(loaded_dataset)):
            data = data_dict[loaded_dataset[i]['name']]
            loaded_dataset[i]['completions']=data['completions']
            loaded_dataset[i]['plans']=data['plans']
            loaded_dataset[i]["passed"]=data["passed"]
            loaded_dataset[i]["pass_num"]=data['pass_num']
            if 'human' in args.dataset:
                loaded_dataset[i]['nl']=data['nl']
                loaded_dataset[i]["func"]=data["func"]
                loaded_dataset[i]["examples"]=data['examples']

        # print(loaded_dataset[0].keys())

        private_tests_dict = {data['name']:data['private_tests'] for data in loaded_dataset} 
        generated_tests_dict = {data['name']:data['generated_tests'] for data in loaded_dataset} 

        for data in loaded_dataset:
            data.pop('private_tests')
            data.pop('generated_tests')
            if 'solutions' in data.keys():
                data.pop('solutions')
        
        # print(loaded_dataset[0].keys())

    prompt_nodes=[]
    passAt10s=[]
    initial_seed_num=-1
    if args.recover<=0:
        
        

        initial_seed = loaded_dataset

        

        


        for i in range(len(loaded_dataset)):
            # print(initial_seed[i])
            if initial_seed[i]['passed']:
                prompt_nodes.append(PromptNode(initial_seed[i],initial_seed[i]['passed'],initial_seed[i]['pass_num']))

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

    for i, prompt_node in enumerate(prompt_nodes):
        prompt_node.index = i
    for prompt_node in prompt_nodes:
        if 'solutions' in prompt_node.solution.keys():
            prompt_node.solution.pop('solutions')
        if 'incorrect_solutions' in prompt_node.solution.keys():
            prompt_node.solution.pop('incorrect_solutions') 
        
    

    num_seed=len(prompt_nodes)

    
    num_seed=len(prompt_nodes)
    # print(initial_seed_num)
    print(f'len of loaded seeds: {initial_seed_num}')


    # print(initial_seed_num)

    invalid_cnt = 0
    
    select_policy = MCTSExploreSelectPolicy(len(prompt_nodes))

    from sentence_transformers import SentenceTransformer, util

    # 加载预训练的Sentence-BERT模型
    if args.calc_analyst==1:
        semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cuda:{}'.format(0))

    # for problem in loaded_dataset:
    #     print(problem['description'])
    #     print(problem['passed'])
    #     print(problem['pass_num'])

    # x=0
    # while True:
    #     if x>1:
    #         break


    fail_list=[]
    threshold=args.num_round - args.recover
    
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



        task=mutated_seed.solution
        intent=task['description']

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
        code_output_path=args.output_path+'code'+'_round_'+str(idx)+'.jsonl'
        with open(code_output_path, 'w+') as f:
            

            # task['private_tests'] = private_tests_dict[task['name']]
            # task['generated_tests'] = generated_tests_dict[task['name']]
            test_cases=''
            if 'mbpp' in args.dataset:
                # print(task.keys())
                test_cases = task['test_list']
            if 'human' in args.dataset:
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
            else:
                try:
                    # 进行分工流程在这里，输入了prompt为intent
                    
                    futures = []
                    # if args.parallel == 1:
                    
                    with ProcessPoolExecutor() as executor:
                        
                        for iteration in range(args.num_generate):
                            cur_problem = copy.deepcopy(task)
                            solver = CodeContestsCompetitor(dataset=args.dataset, method=args.method, model = args.model)
                            ii = (iteration//len(cur_problem['repair_prompt']))%len(cur_problem['repair_prompt'])
                            cur_problem['description'] = cur_problem['repair_prompt'][ii]

                            future= executor.submit(solve_dataset_one_problem_parallel,
                                                    problem = problem, 
                                                    solver = solver,
                                                    dataset_name=args.dataset,
                                                    split_name=args.split_name,
                                                    iteration = iteration)

                            futures.append(future)
                    for _, future in enumerate(as_completed(futures)):
                        # print(future.result())
                        code,plan,passed=future.result()
                        if code == '' and plan == '':
                            continue
                        # if 'mbpp' in args.dataset:
                        #     passed 
                        passing_results.append(passed)
                        codes.append(code)
                        plans.append(plan)
                except Exception as e:
                    print(e)
                    continue
            
                while len(codes)<args.num_generate:
                    ran = random.randint(0,len(codes)-1)
                    codes.append(codes[ran])
                    plans.append(plans[ran])
                    passing_results.append(passing_results[ran])
                
                if True in passing_results:
                    pass_At_10=True
                pass_number=passing_results.count(True)
            
        

                
            # print('-'*10)
            # print(codes)
            # print('-'*10)
            # print(plans)
            # print('-'*10)
            # print(pass_At_10)
            # print('-'*10)
            # print(pass_number)

            task.pop('private_tests')
            task.pop('generated_tests')
            

            task['completions']=codes
            task['plans']=plans
            task['mutate_method'] = cur_mutate_method

            if 'mbpp' in args.dataset:
                score, passes, passAt1, passAt10 = evaluate_one_MBPP(task,args.num_generate)
                pass_At_10,pass_number = passAt10,passes
            if 'human' in args.dataset:
                score, passes, passAt1, passAt10 = evaluate_one(task,args.num_generate)
                pass_At_10,pass_number = passAt10,passes

            mutated_seed.score=pass_At_10
            mutated_seed.passes=pass_number
            
            if mutated_seed.passes<seed.passes:
                task['save_node'] = True
            else:
                task['save_node'] = False
            task['pass'] = pass_At_10
            task['pass_num'] = pass_number
            task['parent_index'] = seed.index
            task['round']=idx
                    
            
            
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
            f.write('seed prompt:\n ' + seed.solution['description']+'\n')
            f.write('mutated prompt:\n' + mutated_seed.solution['description']+'\n\n\n\n')
        

        print('saving......')
        if (idx-1) % 1==0:
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
                'task_id':prompt_node.solution['name'],
                'prompt':prompt_node.solution['description'],
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


    

        
    

