cur_dir=$(cd "$(dirname "$0")"; pwd)
dataset=mbpp
split_name=plus
model=gpt-35-turbo

tag=repair
mutate_method=random
num_round=1000
num_generate=10
run_multi_gen=1
repair_prompt_num=2
repair_plan=1
add_monitor=1
repair_code=1
output_file_name=${tag}_${model}_${dataset}
python /data/zlyuaj/muti-agent/PairCoder/src/main_mix.py \
    --dataset ${dataset} \
    --model ${model} \
    --run_multi_gen ${run_multi_gen}\
    --repair_prompt_num ${repair_prompt_num}\
    --repair_plan ${repair_plan}\
    --add_monitor ${add_monitor}\
    --repair_code ${repair_code}\
    --split_name ${split_name} \
    --dir_path results_${tag} \
    --mutate_method ${mutate_method}\
    --input_path /data/zlyuaj/muti-agent/PairCoder/outputs/fuzzing/results-fuzzing_gpt-35-turbo_mbpp_1-1/_node_1000.jsonl\
    --output_path  ./outputs/${tag}/ \
    --output_file_name ${output_file_name} \
    --save_seed 1\
    --calc_analyst 1\
    --clean_mutate_method 1\
    --split_input 1\
    --alpha 1\
    --num_round ${num_round} \
    --num_generate ${num_generate} | tee ${cur_dir}/${output_file_name}.txt 
