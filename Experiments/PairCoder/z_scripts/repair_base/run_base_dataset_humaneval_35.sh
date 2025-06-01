model=gpt-35-turbo
dataset=humaneval
split_name=plus
tag=repair_basedataset
run_multi_gen=1
repair_prompt_num=2
repair_plan=1
add_monitor=1
repair_code=1
python /data/zlyuaj/muti-agent/PairCoder/src/main_fuzz_passAt10_baseDataset.py \
    --dataset humaneval \
    --split_name ${split_name} \
    --model ${model}\
    --run_multi_gen ${run_multi_gen}\
    --repair_prompt_num ${repair_prompt_num}\
    --repair_plan ${repair_plan}\
    --add_monitor ${add_monitor}\
    --repair_code ${repair_code}\
    --dir_path results \
    --input_path /data/zlyuaj/muti-agent/PairCoder/data/HumanEval_Plus.jsonl\
    --output_path  ./outputs/${tag}/ \
    --output_file_name ${tag}_${dataset}_${model} \
    --num_generate 10\
