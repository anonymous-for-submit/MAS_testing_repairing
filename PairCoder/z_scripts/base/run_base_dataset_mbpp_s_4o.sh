model=gpt-4o
dataset=mbpp
split_name=plus
python /data/zlyuaj/muti-agent/PairCoder/src/main_fuzz_passAt10_baseDataset.py \
    --dataset ${dataset} \
    --split_name ${split_name} \
    --dir_path results \
    --model ${model}\
    --input_path /data/zlyuaj/muti-agent/PairCoder/data/mbpp_sanitized_ET.jsonl\
    --output_path  ./outputs/ \
    --output_file_name ${dataset}_sanitized_${model} \
    --num_generate 10\
