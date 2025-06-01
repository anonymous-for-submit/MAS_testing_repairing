
model=gpt-35-turbo
name_tag=repair_basedataset
dataset=humaneval
num_generate=1
run_generate=1
run_evaluate=1

num_generate=1
run_multi_gen=1
repair_prompt_num=2
repair_plan=1
add_monitor=1
repair_code=1
python /home/zlyuaj/muti-agent/MetaGPT/metagpt/software_main_fuzzing_BaseDataset.py  \
    --model ${model}\
    --run_multi_gen ${run_multi_gen}\
    --repair_prompt_num ${repair_prompt_num}\
    --repair_plan ${repair_plan}\
    --add_monitor ${add_monitor}\
    --repair_code ${repair_code}\
    --input_path /home/zlyuaj/muti-agent/MetaGPT/data/HumanEval_test_case_ET.jsonl \
    --model ${model}\
    --output_path /home/zlyuaj/muti-agent/MetaGPT/output/${name_tag}/ \
    --dataset ${dataset}\
    --output_file_name test \
    --workspace workspace_test_${name_tag}_${model}_${dataset}\
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --parallel 0\
    | tee test.txt  
    # --majority 5 \
    
