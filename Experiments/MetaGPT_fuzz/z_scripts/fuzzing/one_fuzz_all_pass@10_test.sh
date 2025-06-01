cur_dir=$(cd "$(dirname "$0")"; pwd)
dataset=humaneval
model=gpt-35-turbo
output_file_name=test
tag=fuzzing
mutate_method=random
num_round=50
num_generate=2
python ../../metagpt/software_main_fuzzing.py \
    --dataset ${dataset} \
    --model ${model} \
    --mutate_method ${mutate_method}\
    --input_path /home/zlyuaj/muti-agent/MetaGPT/output/basedataset/results-humaneval_gpt-35-turbo/humaneval.jsonl\
    --output_path  ../../output/${tag}/ \
    --output_file_name test \
    --workspace workspace_fuzz_test \
    --save_seed 1\
    --calc_analyst 1\
    --calc_final_result 1\
    --clean_mutate_method 1\
    --split_input 1\
    --parallel 1\
    --alpha 1\
    --mutate_level sentence \
    --num_round ${num_round} \
    --num_generate ${num_generate} | tee test.txt 
