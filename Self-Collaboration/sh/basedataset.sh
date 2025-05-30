
model=gpt-35-turbo
python ../main_baseDataset.py --dataset HumanEval-ET \
    --signature \
    --model ${model} \
    --input_path ../data/HumanEval_test_case_ET.jsonl \
    --output_path ../output/ \
    --do_fuzz 0  \
    --only_consider_passed_cases 0 --mutate_method random \
    --num_generate 10 \
    --clean_data 1 \
    --dataset_type MBPP \
    --num_round 1000 \
    --calc_analyst 1 \
    --output_file_name HumanEvalET_${model} \
    | tee output_HumanEvalET_${model}.txt 
    
