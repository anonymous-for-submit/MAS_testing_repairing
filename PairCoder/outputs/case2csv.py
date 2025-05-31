import pandas as pd
import json
'''
results-analyst_score_1-1_all_plan_random
results-no_save_seed_cleaned_data_random
results-save_seed_by_passes_random
'''
# s=['results-analyst_score_1-1_all_plan_random','results-no_save_seed_cleaned_data_random','results-save_seed_by_passes_random']
s=['results-repair_gpt-35-turbo_analyst_score_1-1']
original_data_path = ''
node_id = 1000
data_path= '/data/zlyuaj/muti-agent/PairCoder/outputs/fuzzing/results-fuzzing_gpt-35-turbo_codecontest_1-1/'
original_data_path = '/data/zlyuaj/muti-agent/PairCoder/outputs/results-codecontest_gpt-35-turbo/codecontest.jsonl'
mas = 'pair'
model = '35'
dataset = 'codecontest'
tag = '1-1'
file_name = f'{mas}_{dataset}_{model}_{tag}' 

original_data_path_GT = ''
if 'human' in dataset:
    original_data_path_GT ='/data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl'
if 'mbpp' in dataset:
    original_data_path_GT ='/data/zlyuaj/muti-agent/fuzzing/data/mbpp_sanitized_ET.jsonl'
if 'codecontest' in dataset:
    original_data_path_GT ='/data/zlyuaj/muti-agent/fuzzing/data/CodeContest_Test.jsonl'


for name in s:
    data_path = data_path.replace('<name>',name)
    cur_data_path=data_path+'_node_{}.jsonl'.format(node_id)



    no_pass=[]
    with open(cur_data_path, 'r') as f:
        datas = [json.loads(line) for line in f]
        for data in datas:
            # if 'score' not in data.keys():
            #     print(data)
            if not data['score']:
                no_pass.append(data['solution'])

    original_dataset=[]
     
    with open(original_data_path, 'r') as f:
        # 导入输出
        original_dataset = [json.loads(line) for line in f] 
    if type(no_pass[0]['name']) == int:
        task_id =[no_pass_data['name'] for no_pass_data in no_pass]
    else:
        task_id = [int(no_pass_data['name'].split('/')[-1]) for no_pass_data in no_pass]
    task_id_str = [no_pass_data['name']for no_pass_data in no_pass]
    prompt = [no_pass_data['description'] for no_pass_data in no_pass]
    task_dict = {task['name']:task for task in original_dataset}
    # print(task_dict)
    # solution['test_case_list'] = task_dict[solution['task_id']]['test_case_list']
    # print(task_id_str[0])
    # print(task_dict[task_id_str[0]].keys())
    if 'mbpp' in dataset or 'human' in dataset:
        original_prompt = [task_dict[task_id]['description'] for task_id in task_id_str]
    if 'codecontest' in dataset:
        original_prompt = [task_dict[task_id]['description'] for task_id in task_id_str]

    # '\n\n'.join() 
    # original_completions = [task_dict[task_id]['completions'] for task_id in task_id_str]
    # self-collab
    original_completions = []
    original_plans=[]
    for task_id in task_id_str:
        pass_completions = []
        pass_plans = []
        if 'completions' not in task_dict[task_id].keys():
            original_completions.append('')
            original_plans.append('')
            continue
        completions = task_dict[task_id]['completions']
        plans = task_dict[task_id]['plans']
        for j in range(len(completions)):
            # if task_dict[task_id]['pass_results'][j]:
                pass_completions.append(completions[j])
                pass_plans.append(plans[j])
        original_completions.append('\n\n'.join(pass_completions))
        original_plans.append('\n\n'.join(pass_plans))
    # original_completions = '\n\n'.join(original_completions)
    # original_plans = '\n\n'.join(original_plans)

   
    # original_plans = ['\n\n'.join([task_dict[task_id]['session_historys'][i]['plan'] for i in range(len(task_dict[task_id]['session_historys']))] )  for task_id in task_id_str]

    with open(original_data_path_GT, 'r') as f:
        # 导入输出
        original_data_GT = [json.loads(line) for line in f] 
        task_dict_GT = {task['task_id']:task for task in original_data_GT}

    if 'human' in dataset:
        GT_solution = [task_dict_GT[task_id]['canonical_solution']  if task_id in task_dict_GT.keys() else '' for task_id in task_id_str]
    if 'mbpp' in dataset :
        GT_solution = [task_dict_GT[int(task_id.split('/')[-1])]['code']  for task_id in task_id_str]
    if 'codecontest' in dataset:
        GT_solution = ['']*len(original_prompt)

        
    if 'entry_point' not in task_dict[task_id_str[0]].keys():
        entry_point = ['']*len(original_prompt)
    else:
        entry_point = [task_dict[task_id]['entry_point']  if 'entry_point' in task_dict[task_id].keys() else '' for task_id in task_id_str]
    # GT_solution = [original_dataset[i]['entry_point'] for i in task_id]
    completions = [no_pass_data['completions'][0] for no_pass_data in no_pass]
    plans = [no_pass_data['plans'][0] for no_pass_data in no_pass]
    # entry_point = [original_dataset[i]['entry_point'] for i in tas`k_id]
    # plan_completions=[]
    # for i in range(1):
    #     plan_completion = [plans[i][j]+'\n####\n'+completions[i][j] for j in range(10)]
    #     plan_completions.append(plan_completion)


    #任意的多组列表
    a = [1,2,3]
    b = [4,5,6]    

    #字典中的key值即为csv中列名
    # import pandas as pd
    dataframe = pd.DataFrame({'task_id':task_id,'entry_point':entry_point,'prompt':prompt,'original_prompt':original_prompt,'GT_solution':GT_solution,'plan':plans,'completion':completions, 'original_plans': original_plans, 'original_completions':original_completions })

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    
    dataframe.to_csv("{}.csv".format(file_name),index=False,sep=',')
