cur_dir=$(cd "$(dirname "$0")"; pwd)
dataset=humaneval
model=gpt-35-turbo
tag=repair_fuzzing
mutate_method=random
num_round=1000
num_generate=10
run_multi_gen=1
repair_prompt_num=2
repair_plan=1
add_monitor=1
repair_code=1
python ../../metagpt/software_main_fuzzing.py  \
    --dataset ${dataset} \
    --model ${model} \
    --run_multi_gen ${run_multi_gen}\
    --repair_prompt_num ${repair_prompt_num}\
    --repair_plan ${repair_plan}\
    --add_monitor ${add_monitor}\
    --repair_code ${repair_code}\
    --mutate_method ${mutate_method}\
    --input_path /home/zlyuaj/muti-agent/MetaGPT/output/basedataset/results-humaneval_gpt-35-turbo/humaneval.jsonl\
    --output_path  ../../output/${tag}/ \
    --output_file_name ${tag}_${model}_${dataset}_1-1 \
    --workspace workspace_${tag}_${model}_${dataset} \
    --save_seed 1\
    --calc_analyst 1\
    --calc_final_result 1\
    --clean_mutate_method 1\
    --split_input 1\
    --parallel 1\
    --alpha 1\
    --mutate_level sentence \
    --num_round ${num_round} \
    --num_generate ${num_generate} | tee ${cur_dir}/${tag}_${model}_${dataset}_1-1.txt 
