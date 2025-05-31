model=deepseek-coder
dataset=humaneval
split_name=plus
python /data/zlyuaj/muti-agent/PairCoder/src/main_fuzz_passAt10_baseDataset.py \
    --dataset humaneval \
    --split_name ${split_name} \
    --model ${model}\
    --dir_path results \
    --input_path /data/zlyuaj/muti-agent/PairCoder/data/HumanEval_test_case_ET.jsonl\
    --output_path  ./outputs/ \
    --output_file_name ${dataset}_et_${model} \
    --num_generate 10\
    --start_idx 61\
