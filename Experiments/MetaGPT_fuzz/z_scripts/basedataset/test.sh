
model=gpt-4o
name_tag=basedataset
dataset=codecontest
num_generate=1
run_generate=1
run_evaluate=1
python /home/zlyuaj/muti-agent/MetaGPT/metagpt/software_main_fuzzing_BaseDataset.py  \
    --model ${model}\
    --input_path /home/zlyuaj/muti-agent/MetaGPT/data/CodeContest_Test.jsonl \
    --model ${model}\
    --output_path /home/zlyuaj/muti-agent/MetaGPT/output/${name_tag}/ \
    --dataset ${dataset}\
    --output_file_name test \
    --workspace workspace_${model}_${dataset}_test \
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --parallel 0\
    | tee test.txt  
    # --majority 5 \
    
