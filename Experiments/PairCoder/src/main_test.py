from code_contests.data.provider import CodeContestDataProvider
from gen.coding_competitor import CodeContestsCompetitor
from gen.utils import evaluate_solution_on_subset
from log import setup_logger
from settings.config_loader import get_settings
from gen.dataset_solver import solve_dataset,solve_dataset_one_problem,solve_dataset_one_problem_parallel
import toml
import datetime
if __name__ == '__main__':


    from datasets import load_dataset
    # ds = load_dataset("dz1/CodeScore-MBPP-ET")



    dataset = 'mbpp'
    split_name = 'plus'
    data_provider = CodeContestDataProvider(dataset_location=dataset)
    num_problems = len(data_provider.dataset[split_name])

    loaded_dataset = []
    for problem_number in range(5):
        problem_name = data_provider.dataset[split_name][int(problem_number)]['name']

        # 从导入的dataset里面找到问题
        problem = data_provider.find_problem(ds=data_provider.dataset, problem_name=problem_name, split_name=split_name)
        problem['dataset_name'] =dataset
        print(type(problem))
        for key in problem.keys():
            if key=='public_tests':
                print(key)
                print(problem[key])
        pt = problem['private_tests']
        inputs =pt['input']
        outputs = pt['output']
        # for i in range(len(inputs)):
        #     print('-'*20)
        #     print(inputs[i])
        #     print(outputs[i])
        loaded_dataset.append(problem)
    


