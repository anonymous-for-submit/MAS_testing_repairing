F='/data/zlyuaj/muti-agent/MetaGPT/output/results-test_3/HumanEval_ET.jsonl'
import json
with open(F,'r') as f:
    lines=[json.loads(line) for line in f]
    for line in lines:
        print('#'*50)
        print(line['prompt'])
        for c in line['completions']:
            print('-'*50)
            print(c)
        x=input()
