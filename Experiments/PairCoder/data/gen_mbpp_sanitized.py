
import os
import json
file_dir = '/data/zlyuaj/muti-agent/fuzzing/data/mbpp_sanitized/'
files = os.listdir(file_dir)
read=False
merge=False
add_test = True
if read:
    for file in files:
        import pyarrow.parquet as pq

        # 读取Parquet文件
        print(file)
        table = pq.read_table(file_dir+file)

        import pandas as pd

        # 将Parquet数据转换为DataFrame
        df = table.to_pandas()
        # 将DataFrame转换为JSON格式
        json_data = df.to_json(orient='records', lines=True)
        # 将JSON数据写入文件
        with open('mbpp_sanitized_{}.jsonl'.format(file.split('-')[0]), 'w') as f:
            f.write(json_data)
if merge:
    dataset = []
    for file in files:
        with open('mbpp_sanitized_{}.jsonl'.format(file.split('-')[0]), 'r') as f:
            dataset+=[json.loads(line) for line in f]
    with open('mbpp_sanitized.jsonl', 'w') as f:
        for t in dataset:
            f.write(json.dumps(t)+'\n')

if add_test:
    dataset=[]
    dataset_test_list=[]
    test_dict={}
    with open('mbpp_sanitized.jsonl', 'r') as f:
        dataset=[json.loads(line) for line in f]
    with open('MBPP_ET.jsonl', 'r') as f:
        dataset_test_list=[json.loads(line) for line in f]
        test_dict = {task['task_id']:task for task in dataset_test_list}
    for i in range(len(dataset)):
        dataset[i]['test_list'] = test_dict[dataset[i]['task_id']]['test_list']
        dataset[i]['entry_point'] = test_dict[dataset[i]['task_id']]['entry_point']
    with open('mbpp_sanitized_ET.jsonl', 'w') as f:
        for t in dataset:
            f.write(json.dumps(t)+'\n')