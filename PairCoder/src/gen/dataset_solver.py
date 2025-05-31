import json
import os
from collections import OrderedDict
from concurrent.futures import as_completed, ProcessPoolExecutor
from code_contests.data.provider import CodeContestDataProvider
from gen.coding_competitor import CodeContestsCompetitor
from gen.utils import evaluate_solution_on_subset
from log import setup_logger
from settings.config_loader import get_settings
from main_mutate import get_more_prompt_test
import toml
import copy
def solve_dataset_one_problem_parallel(problem,
                                       solver,
                                dataset_name='codecontest',
                                split_name='valid',
                                iteration = 0,
                                repair_prompt = False,
                                repair_code = False,
                                add_monitor=False,
                                ):
    setting = get_settings()
    base_path = os.getcwd()
    setting.solve.reduce_verbose = True
    # 建立log位置和输出位置·

    log_root = f'./logs/{dataset_name}/{split_name}/{setting.get("config.model", "PATH_ERROR")}'

    os.makedirs(log_root, exist_ok=True)
    if log_root[-1]=='/':
        log_root = log_root[:-1]

    problem_number = problem['name']
    if 'human' in dataset_name.lower():
        problem_number = int(problem_number.split('/')[1])

    logger = setup_logger(logger_path=f"{log_root}/{split_name}_{problem_number}.log",level='FATAL')
    
    
    if 'humaneval' in dataset_name.lower() or 'mbpp' in dataset_name.lower():
        problem['io_format'] = 'normal'
    else:
        problem['io_format'] = 'contest'

    problem_database = {problem_number: {}}


    it_str = f"iteration_{iteration}"
    problem_database[problem_number][it_str] = {}

    if 'human' in dataset_name.lower():
        if not problem['public_tests']['input'] and not problem['public_tests']['output']:
            logger.info(f"There is no public tests in {problem['name']}, use the first private test!")
            problem['public_tests']['input'] = [problem['private_tests']['input'][0]]
            problem['public_tests']['output'] = [problem['private_tests']['output'][0]]

    predictions = solver.solve_problem_in_dataset(problem, iteration)
    if len(predictions)!=2:
        logger.info(f"Failed to solve problem {problem_number} in iteration {iteration}")
        return '','',False
    
    [solution,plan] = predictions
    logger.info(f"solution code:\n{solution}")
    logger.info(f"Evaluating solution on public tests...")
    silent = True
    test_results, test_passed_public, test_failed_public, test_timeout_public = evaluate_solution_on_subset(
        'public_tests', problem, solution, silent=silent, only_failed_cases=True)
    
    problem_database[problem_number][it_str]['solution'] = solution
    problem_database[problem_number][it_str]['test_passed_public'] = test_passed_public
    problem_database[problem_number][it_str]['test_failed_public'] = test_failed_public
    problem_database[problem_number][it_str]['test_timeout_public'] = test_timeout_public
    # if not is_iterative_method(method):
    #     if not is_solved(problem_database[problem_number][it_str], domain="public"):
    #         if iteration == num_iterations-1:
    #             logger.info(f"Failed to solve problem {problem_number}'s public cases in all {setting.get('dataset.num_iterations', 1)} iterations. Submit it!")
    #         else:
    #             logger.info(f"Failed to solve problem {problem_number}'s public cases in iteration {iteration}")
    #             logger.info(f"\ntest_passed_public: {test_passed_public}, test_failed_public: {test_failed_public}, test_timeout_public: {test_timeout_public}\n")
    #             continue
    #     else:
    #         logger.info(f"solved problem {problem_number}'s public cases in iteration {iteration}. Submit it!")

    logger.info(f"evaluating solution on private tests...")
    # evaluate
    test_results, test_passed_private, test_failed_private, test_timeout_private = evaluate_solution_on_subset(
        'private_tests', problem, solution, silent=silent, only_failed_cases=True)

    logger.info(f"evaluating solution on generated tests...")
    test_results, test_passed_generate, test_failed_generate, test_timeout_generate = evaluate_solution_on_subset(
        'generated_tests', problem, solution, silent=silent, only_failed_cases=True)

    logger.info(
        f"\ntest_passed_public: {test_passed_public}, test_failed_public: {test_failed_public}, test_timeout_public: {test_timeout_public}\n"
        f"test_passed_private: {test_passed_private}, test_failed_private: {test_failed_private}, test_timeout_private: {test_timeout_private}\n"
        f"test_passed_generate: {test_passed_generate}, test_failed_generate: {test_failed_generate}, test_timeout_generate: {test_timeout_generate}\n")

    problem_database[problem_number][it_str]['test_passed_private'] = test_passed_private
    problem_database[problem_number][it_str]['test_failed_private'] = test_failed_private
    problem_database[problem_number][it_str]['test_timeout_private'] = test_timeout_private
    problem_database[problem_number][it_str]['test_passed_generate'] = test_passed_generate
    problem_database[problem_number][it_str]['test_failed_generate'] = test_failed_generate
    problem_database[problem_number][it_str]['test_timeout_generate'] = test_timeout_generate
    problem_database[problem_number][it_str]['test_passed_public'] = test_passed_public
    problem_database[problem_number][it_str]['test_failed_public'] = test_failed_public
    problem_database[problem_number][it_str]['test_timeout_public'] = test_timeout_public
    problem_database[problem_number][it_str]['plan']=plan
    os.chdir(base_path)
    if is_solved(problem_database[problem_number][it_str]):
        logger.info(f"PairCoder solved problem {problem_number} in iteration {iteration}")
    else:
        logger.info(f"PairCoder failed to solve problem {problem_number} in iteration {iteration}")


    itr_result_dict = problem_database[problem_number][it_str]

    passed = is_solved(itr_result_dict)
    code = itr_result_dict['solution']
    plan = itr_result_dict['plan']

    return code,plan,passed



        

    
def solve_dataset_one_problem(problem,
                  dataset_name='codecontest',
                  split_name='valid',
                  solution_file_name='solutions.json',
                  num_iterations = 1,
                  dir_path=None,
                  method=None,
                  model = 'gpt-35-turbo',
                  repair_plan = False,
                    repair_code = False,
                    add_monitor=False,
                  ):
    # 导入数据


    
    setting = get_settings()
    base_path = os.getcwd()
    setting.solve.reduce_verbose = True
    # 建立log位置和输出位置·

    log_root = f'./logs/{dataset_name}/{split_name}/{setting.get("config.model", "PATH_ERROR")}'
    if dir_path:
        log_root += f'/{dir_path}/'
    os.makedirs(log_root, exist_ok=True)
    if log_root[-1]=='/':
        log_root = log_root[:-1]
    
    problem_number = problem['name']
    if 'human' in dataset_name.lower() or 'mbpp' in dataset_name.lower():
        problem_number = int(problem_number.split('/')[1])
    
    
    logger = setup_logger(logger_path=f"{log_root}/{split_name}_{problem_number}.log",level='FATAL')
    # logger = setup_logger(logger_path=f"{log_root}/{split_name}_{problem_number}.log")

    os.chdir(base_path)
    
    database = {split_name: {}}

    
    
    if 'humaneval' in dataset_name.lower() or 'mbpp' in dataset_name.lower():
        problem['io_format'] = 'normal'
    else:
        problem['io_format'] = 'contest'
    problem_database = {problem_number: {}}
    # return 
    
    # num_iterations =1
    solver = CodeContestsCompetitor(dataset=dataset_name, method=method, model = model,repair_plan = repair_plan,repair_code = repair_code,add_monitor=add_monitor)

    for iteration in range(num_iterations):
        ii = (iteration//len(problem['repair_prompt']))%len(problem['repair_prompt'])
        problem['description'] = problem['repair_prompt'][ii]
        it_str = f"iteration_{iteration}"
        problem_database[problem_number][it_str] = {}
        # prev_iter = database[split_name].get(str(problem_number), {}).get(it_str, {})
        # if not ((prev_iter == {}) or (prev_iter is None)):
        #     print(f"prev_iter {iteration} already ran")
        #     problem_database[problem_number][it_str] = prev_iter
        #     if is_solved(prev_iter):
        #         logger.info(f"solved problem {problem_number}")
        #         break
        #     continue
        max_try=1
        while True:
            # if 'mbpp' in dataset_name:
            #     index = problem['description'].find('assert')
            #     if index== -1:
            #         example_str = '\nExample:\n' + problem['test_list'][0].replace('\"', "'")
            #         problem['description'] += example_str
            #         if problem['test_setup_code']:
            #             problem['description'] += '\nSetup Code:\n' + problem['test_setup_code']
            #     else:
            #         example_str = problem['description'][index:]
            #         problem['description'] += 'Here are some public test cases:'
            #         for i, case in enumerate(zip(problem['public_tests']['input'], problem['public_tests']['output'])):
            #             problem['description'] += f'\nExample{i}:\n' + f'  Input: {case[0]}\n' + f'  Output: {case[1]}'
            if 'human' in dataset_name.lower():
                if not problem['public_tests']['input'] and not problem['public_tests']['output']:
                    logger.info(f"There is no public tests in {problem['name']}, use the first private test!")
                    problem['public_tests']['input'] = [problem['private_tests']['input'][0]]
                    problem['public_tests']['output'] = [problem['private_tests']['output'][0]]
            # solve的核心函数
            predictions = solver.solve_problem_in_dataset(problem, iteration)
            if len(predictions)!=2:
                max_try-=1
                if max_try<=0:
                    break
                continue
            
            [solution,plan] = predictions
            logger.info(f"solution code:\n{solution}")
            if not solution:
                logger.info(f"Failed to solve problem {problem_number} in iteration {iteration}")
                continue
            logger.info(f"Evaluating solution on public tests...")
            silent = True
            test_results, test_passed_public, test_failed_public, test_timeout_public = evaluate_solution_on_subset(
                'public_tests', problem, solution, silent=silent, only_failed_cases=True)
            
            problem_database[problem_number][it_str]['solution'] = solution
            problem_database[problem_number][it_str]['test_passed_public'] = test_passed_public
            problem_database[problem_number][it_str]['test_failed_public'] = test_failed_public
            problem_database[problem_number][it_str]['test_timeout_public'] = test_timeout_public
            if not is_iterative_method(method):
                if not is_solved(problem_database[problem_number][it_str], domain="public"):
                    if iteration == num_iterations-1:
                        logger.info(f"Failed to solve problem {problem_number}'s public cases in all {setting.get('dataset.num_iterations', 1)} iterations. Submit it!")
                    else:
                        logger.info(f"Failed to solve problem {problem_number}'s public cases in iteration {iteration}")
                        logger.info(f"\ntest_passed_public: {test_passed_public}, test_failed_public: {test_failed_public}, test_timeout_public: {test_timeout_public}\n")
                        continue
                else:
                    logger.info(f"solved problem {problem_number}'s public cases in iteration {iteration}. Submit it!")

            logger.info(f"evaluating solution on private tests...")
            # evaluate
            test_results, test_passed_private, test_failed_private, test_timeout_private = evaluate_solution_on_subset(
                'private_tests', problem, solution, silent=silent, only_failed_cases=True)

            logger.info(f"evaluating solution on generated tests...")
            test_results, test_passed_generate, test_failed_generate, test_timeout_generate = evaluate_solution_on_subset(
                'generated_tests', problem, solution, silent=silent, only_failed_cases=True)

            logger.info(
                f"\ntest_passed_public: {test_passed_public}, test_failed_public: {test_failed_public}, test_timeout_public: {test_timeout_public}\n"
                f"test_passed_private: {test_passed_private}, test_failed_private: {test_failed_private}, test_timeout_private: {test_timeout_private}\n"
                f"test_passed_generate: {test_passed_generate}, test_failed_generate: {test_failed_generate}, test_timeout_generate: {test_timeout_generate}\n")

            problem_database[problem_number][it_str]['test_passed_private'] = test_passed_private
            problem_database[problem_number][it_str]['test_failed_private'] = test_failed_private
            problem_database[problem_number][it_str]['test_timeout_private'] = test_timeout_private
            problem_database[problem_number][it_str]['test_passed_generate'] = test_passed_generate
            problem_database[problem_number][it_str]['test_failed_generate'] = test_failed_generate
            problem_database[problem_number][it_str]['test_timeout_generate'] = test_timeout_generate
            problem_database[problem_number][it_str]['test_passed_public'] = test_passed_public
            problem_database[problem_number][it_str]['test_failed_public'] = test_failed_public
            problem_database[problem_number][it_str]['test_timeout_public'] = test_timeout_public
            problem_database[problem_number][it_str]['plan']=plan
            os.chdir(base_path)
            if is_solved(problem_database[problem_number][it_str]):
                logger.info(f"PairCoder solved problem {problem_number} in iteration {iteration}")
            else:
                logger.info(f"PairCoder failed to solve problem {problem_number} in iteration {iteration}")
            max_try-=1
            if problem_database[problem_number][it_str] or max_try<=0:
                break

            
    database[split_name][problem_number] = problem_database[problem_number]
    os.chdir(base_path)
    with open(solution_file_name, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=4)

    # 复原description
    problem['description'] = problem['repair_prompt'][0]
    
    codes = []
    plans = []
    passing_results=[]
    pass_At_10 = False
    pass_number = 0
    result_dict = problem_database[problem_number]
    for iteration in range(num_iterations):
        it_str = f'iteration_{iteration}'
        itr_result_dict = result_dict[it_str]
        if not itr_result_dict:
            continue
        codes.append(itr_result_dict['solution'])
        plans.append(itr_result_dict['plan'])
        passing_results.append(is_solved(itr_result_dict))

    import random
    if not codes:
        codes = ['']
        plans = ['']
        passing_results = [False]
    while len(codes) < num_iterations:
        ran = random.randint(0,len(codes)-1)
        codes.append(codes[ran])
        plans.append(plans[ran])
        passing_results.append(passing_results[ran])
                
    if True in passing_results:
        pass_At_10=True
    pass_number=passing_results.count(True)





    return [codes,plans,pass_At_10,pass_number]


def solve_dataset(args,dataset_name='codecontest',
                  split_name='valid',
                  solution_file_name='solutions.json',
                  num_iterations = 1,
                  id_range=None,
                  id_list=None,
                  dir_path=None,
                  method=None,
                  model = 'gpt-35-turbo',
                  repair_plan = False,
                    repair_code = False,
                    add_monitor=False,
                    

                  ):
    # 导入数据
    data_provider = CodeContestDataProvider(dataset_location=dataset_name)
    num_problems = len(data_provider.dataset[split_name])
    print(f'num_problems: {num_problems}')


    setting = get_settings()
    base_path = os.getcwd()
    setting.solve.reduce_verbose = True
    # 建立log位置和输出位置·
    log_root = f'./logs/{dataset_name}/{split_name}/{setting.get("config.model", "PATH_ERROR")}'
    if dir_path:
        log_root += f'/{dir_path}/'
    os.makedirs(log_root, exist_ok=True)
    if len(solution_file_name) == 0:
        solution_file_name = 'solutions.json'
    solution_file_name = f'{log_root}/{solution_file_name}'
    config_dict = setting.to_dict()
    with open(f'{log_root}/config.toml', 'w') as toml_file:
        toml.dump(config_dict, toml_file)
    # try:
    #     with open(solution_file_name, 'r') as f:
    #         database = json.load(f)
    #         database[split_name] = OrderedDict(sorted(database[split_name].items(), key=lambda x: int(x[0])))
    # except:
    #     print(f"Failed to load database from {solution_file_name}")
    database = {split_name: {}}
    
    # num_problems=1
    # all_problems=[]
    for problem_number in range(0, num_problems):
        # 可以手动设置要跑哪些函数
        if id_range is not None:
            id_num = problem_number
            low, high = id_range
            if id_num < low or id_num >= high:
                print(f"Skipping {problem_number} as it is not in {id_range}")
                continue
        if id_list and problem_number not in id_list:
            continue
        logger = setup_logger(logger_path=f"{log_root}/{split_name}_{problem_number}.log",level='WARNING')
        # num_iterations =  setting.get("dataset.num_iterations", 1)
        prev = database[split_name].get(str(problem_number), {})
        # if not ((prev == {}) or (prev is None)):
        #     print(f"problem_number {problem_number} already ran")
        #     continue

        os.chdir(base_path)
        logger.info(f"problem_number: {problem_number}")
        problem_name = data_provider.dataset[split_name][int(problem_number)]['name']
        logger.info(f"problem_name: {problem_name}")
        # 从导入的dataset里面找到问题
        problem = data_provider.find_problem(ds=data_provider.dataset, problem_name=problem_name, split_name=split_name)
        

        # multi gen
        intent=problem['description']
        repair_prompt=[]
        mutation_method = ['expand_one2two','condense_two2one','rephrase_one']
        if args.run_multi_gen==1:
            for i in range(args.repair_prompt_num):
                new_prompt= get_more_prompt_test(intent,args,mutation_method[i])
                print('multi-gen-prompt:')
                print(new_prompt)
                repair_prompt.append(new_prompt)
        problem['repair_prompt'] = [intent]+repair_prompt
        
        problem['dataset_name'] = dataset_name

        

       
        # for key,value in problem.items():
        #     print(key)
        #     print(value)
        # print(problem['description'])
        # print(problem['private_tests']['input'][:5])
        # print(problem['private_tests']['output'][:5])
        # return ['']*4
    
        
        if 'humaneval' in dataset_name.lower() or 'mbpp' in dataset_name.lower():
            problem['io_format'] = 'normal'
        else:
            problem['io_format'] = 'contest'
        problem_database = {problem_number: {}}
        # return 
        solver = CodeContestsCompetitor(dataset=dataset_name, method=method, model = model,repair_plan = repair_plan,repair_code = repair_code,add_monitor=add_monitor)
        # num_iterations =1
        
        for iteration in range(num_iterations):
            it_str = f"iteration_{iteration}"
            ii = (iteration//len(problem['repair_prompt']))%len(problem['repair_prompt'])
            problem['description'] = problem['repair_prompt'][ii]
            problem_database[problem_number][it_str] = {}
            # prev_iter = database[split_name].get(str(problem_number), {}).get(it_str, {})
            # if not ((prev_iter == {}) or (prev_iter is None)):
            #     print(f"prev_iter {iteration} already ran")
            #     problem_database[problem_number][it_str] = prev_iter
            #     if is_solved(prev_iter):
            #         logger.info(f"solved problem {problem_number}")
            #         break
            #     continue
            max_try=3
            while True:
                # if 'mbpp' in dataset_name:
                #     index = problem['description'].find('assert')
                #     if index== -1:
                #         example_str = '\nExample:\n' + problem['test_list'][0].replace('\"', "'")
                #         problem['description'] += example_str
                #         if problem['test_setup_code']:
                #             problem['description'] += '\nSetup Code:\n' + problem['test_setup_code']
                #     else:
                #         example_str = problem['description'][index:]
                #         problem['description'] += 'Here are some public test cases:'
                #         for i, case in enumerate(zip(problem['public_tests']['input'], problem['public_tests']['output'])):
                #             problem['description'] += f'\nExample{i}:\n' + f'  Input: {case[0]}\n' + f'  Output: {case[1]}'
                if 'human' in dataset_name.lower():
                    if not problem['public_tests']['input'] and not problem['public_tests']['output']:
                        logger.info(f"There is no public tests in {problem['name']}, use the first private test!")
                        problem['public_tests']['input'] = [problem['private_tests']['input'][0]]
                        problem['public_tests']['output'] = [problem['private_tests']['output'][0]]
                # solve的核心函数
                predictions = solver.solve_problem_in_dataset(problem, iteration)
                if len(predictions)!=2:
                    max_try-=1
                    if max_try<=0:
                        break
                    continue
                
                [solution,plan] = predictions
                logger.info(f"solution code:\n{solution}")
                if not solution:
                    logger.info(f"Failed to solve problem {problem_number} in iteration {iteration}")
                    continue
                logger.info(f"Evaluating solution on public tests...")
                silent = True
                test_results, test_passed_public, test_failed_public, test_timeout_public = evaluate_solution_on_subset(
                    'public_tests', problem, solution, silent=silent, only_failed_cases=True)
                
                problem_database[problem_number][it_str]['solution'] = solution
                problem_database[problem_number][it_str]['test_passed_public'] = test_passed_public
                problem_database[problem_number][it_str]['test_failed_public'] = test_failed_public
                problem_database[problem_number][it_str]['test_timeout_public'] = test_timeout_public
                if not is_iterative_method(method):
                    if not is_solved(problem_database[problem_number][it_str], domain="public"):
                        if iteration == num_iterations-1:
                            logger.info(f"Failed to solve problem {problem_number}'s public cases in all {setting.get('dataset.num_iterations', 1)} iterations. Submit it!")
                        else:
                            logger.info(f"Failed to solve problem {problem_number}'s public cases in iteration {iteration}")
                            logger.info(f"\ntest_passed_public: {test_passed_public}, test_failed_public: {test_failed_public}, test_timeout_public: {test_timeout_public}\n")
                            continue
                    else:
                        logger.info(f"solved problem {problem_number}'s public cases in iteration {iteration}. Submit it!")

                logger.info(f"evaluating solution on private tests...")
                # evaluate
                test_results, test_passed_private, test_failed_private, test_timeout_private = evaluate_solution_on_subset(
                    'private_tests', problem, solution, silent=silent, only_failed_cases=True)

                logger.info(f"evaluating solution on generated tests...")
                test_results, test_passed_generate, test_failed_generate, test_timeout_generate = evaluate_solution_on_subset(
                    'generated_tests', problem, solution, silent=silent, only_failed_cases=True)

                logger.info(
                    f"\ntest_passed_public: {test_passed_public}, test_failed_public: {test_failed_public}, test_timeout_public: {test_timeout_public}\n"
                    f"test_passed_private: {test_passed_private}, test_failed_private: {test_failed_private}, test_timeout_private: {test_timeout_private}\n"
                    f"test_passed_generate: {test_passed_generate}, test_failed_generate: {test_failed_generate}, test_timeout_generate: {test_timeout_generate}\n")

                problem_database[problem_number][it_str]['test_passed_private'] = test_passed_private
                problem_database[problem_number][it_str]['test_failed_private'] = test_failed_private
                problem_database[problem_number][it_str]['test_timeout_private'] = test_timeout_private
                problem_database[problem_number][it_str]['test_passed_generate'] = test_passed_generate
                problem_database[problem_number][it_str]['test_failed_generate'] = test_failed_generate
                problem_database[problem_number][it_str]['test_timeout_generate'] = test_timeout_generate
                problem_database[problem_number][it_str]['test_passed_public'] = test_passed_public
                problem_database[problem_number][it_str]['test_failed_public'] = test_failed_public
                problem_database[problem_number][it_str]['test_timeout_public'] = test_timeout_public
                problem_database[problem_number][it_str]['plan']=plan
                os.chdir(base_path)
                if is_solved(problem_database[problem_number][it_str]):
                    logger.info(f"PairCoder solved problem {problem_number} in iteration {iteration}")
                else:
                    logger.info(f"PairCoder failed to solve problem {problem_number} in iteration {iteration}")
                max_try-=1
                if problem_database[problem_number][it_str] or max_try<=0:
                    break
        database[split_name][problem_number] = problem_database[problem_number]
        os.chdir(base_path)
        with open(solution_file_name, 'w', encoding='utf-8') as f:
            json.dump(database, f, ensure_ascii=False, indent=4)

        codes = []
        plans = []
        pass_At_10 = False
        pass_number = 0
        result_dict = problem_database[problem_number]
        for iteration in range(num_iterations):
            it_str = f'iteration_{iteration}'
            itr_result_dict = result_dict[it_str]
            if not itr_result_dict:
                print(f'There is still no solution for iteration {iteration}! Using the first one!')
                if iteration>0:
                    itr_result_dict = result_dict['iteration_0']
                
            if not itr_result_dict:
                codes.append('')
                plans.append('')
            else:
                codes.append(itr_result_dict['solution'])
                plans.append(itr_result_dict['plan'])
                if is_solved(itr_result_dict):
                    pass_number+=1
                    pass_At_10=True
        return [codes,plans,pass_At_10,pass_number,problem['repair_prompt']]

        



def is_iterative_method(method):
    if method in ['direct', 'cot', 'scot', 'self_planning']:
        return False
    else:
        return True

def is_solved(s, domain='all'):
    if domain == 'public':
        if s['test_failed_public'] == 0 and \
            s['test_timeout_public'] == 0 and \
            s['test_passed_public'] > 0:
            return True
        else:
            return False
    else:
        if s['test_failed_private'] == 0 and s['test_failed_generate'] == 0 and \
                s['test_timeout_private'] == 0 and s['test_timeout_generate'] == 0 and \
                (s['test_passed_private'] + s['test_passed_generate']) > 0:
            return True
        else:
            return False

'''
 if 'human' in problem['dataset_name']:
            et_data_path = './data/HumanEval_test_case_ET.jsonl'
        elif 'mbpp' in problem['dataset_name']:
            et_data_path = './data/mbpp_sanitized_ET.jsonl'
        else:
            et_data_path = ''
        # assert has_close_elements([4.88, 7.89, 3.67, 5.68, 4.88], 2.06) == True
        def add_test_case(problem,et_test_dict):
            inputs = []
            outputs = []
            test_list = et_test_dict[problem['name']]
            for test in test_list:
                if ' == ' in test:
                    [input,output]= test.split(' == ')
                    input = input[input.find('(')+1:input.find(')')]
                    inputs.append(input)
                    outputs.append(output)
                elif 'abs' in test and ' 1e-' in test:
                    test = test[test.find('(')+1:test.find(')')]
                    if ' - ' in test:
                        [input,output]= test.split(' - ')
                        input = input[input.find('(')+1:input.find(')')]
                        # print(output)
                        # print('.' in output)
                        if '.' in output:
                            # print(right_count)
                            right_count= len(output.split('.')[1])
                            output+='0'*(17-right_count)
                        inputs.append(input)
                        outputs.append(output)
            if inputs and outputs:
                problem['private_tests']['input'] = inputs
                problem['private_tests']['output'] = outputs
            return problem

        add_test=False
        if et_data_path and add_test:
            sanitized_task_list = []
            et_test_dict={}
            with open(et_data_path, 'r') as f:
                loaded_dataset = [json.loads(line) for line in f]
                original_task_list = [data['task_id'] for data in loaded_dataset]
                # human_id_list = [data['task_id'] for data in loaded_dataset]
                if 'human' in problem['dataset_name']:
                    et_test_dict = {data['task_id']:data['test_case_list'] for data in loaded_dataset}
                elif 'mbpp' in problem['dataset_name']:
                    et_test_dict = {data['task_id']:data['test_list'] for data in loaded_dataset}
            if 'mbpp' in problem['dataset_name'] and problem['name'] not in original_task_list:
                print('mbpp problem {} not in sanitized_task_list'.format(problem['name']))
                continue
            if 'human' in problem['dataset_name'] and problem['name'] not in original_task_list:
                print('humaneval problem {} not in human_id_list'.format(problem['name']))
                continue

            problem = add_test_case(problem,et_test_dict)
'''