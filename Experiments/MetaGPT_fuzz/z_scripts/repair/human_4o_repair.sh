cur_dir=$(cd "$(dirname "$0")"; pwd)
model=gpt-4o
name_tag=repair
dataset=humaneval
num_generate=1
run_generate=1
run_evaluate=1
output_file_name=test
tag=repair

num_generate=10
run_multi_gen=1
repair_prompt_num=2
repair_plan=1
add_monitor=1
repair_code=1
python ../../metagpt/software_main_mix.py \
    --dataset ${dataset} \
    --model ${model} \
    --run_multi_gen ${run_multi_gen}\
    --repair_prompt_num ${repair_prompt_num}\
    --repair_plan ${repair_plan}\
    --add_monitor ${add_monitor}\
    --repair_code ${repair_code}\
    --input_path ../../output/fuzzing/results-fuzzing_${model}_${dataset}_1-1/_node_1000.jsonl\
    --output_path  ../../output/${tag}/ \
    --output_file_name ${output_file_name} \
    --workspace workspace_repair_${model}_${dataset} \
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --parallel 1\ | tee ${cur_dir}/${tag}_${model}_${dataset}.txt 
