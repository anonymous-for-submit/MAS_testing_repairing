cur_dir=$(cd "$(dirname "$0")"; pwd)
dataset=codecontest
model=gpt-4o
tag=fuzzing
mutate_method=random
num_round=1000
num_generate=10
python ../../metagpt/software_main_fuzzing.py  \
    --dataset ${dataset} \
    --model ${model} \
    --mutate_method ${mutate_method}\
    --input_path /home/zlyuaj/muti-agent/MetaGPT/output/basedataset/results-codecontest_gpt-4o/codecontest.jsonl\
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
    --max_visit_num 15\
    --num_generate ${num_generate} | tee ${cur_dir}/${tag}_${model}_${dataset}_1-1.txt 
