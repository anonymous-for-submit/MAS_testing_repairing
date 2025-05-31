
model=gpt-35-turbo
repair_prompt_num=1
python ../main_fuzz_passAt10_baseDataset.py --dataset HumanEval-ET \
    --signature \
    --model ${model} \
    --input_path /data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl \
    --output_path /data/zlyuaj/muti-agent/fuzzing/output_mutated/original/ \
    --do_fuzz 0  \
    --only_consider_passed_cases 0 --mutate_method random \
    --num_generate 10 \
    --clean_data 1 \
    --dataset_type humaneval \
    --num_round 1000 \
    --calc_analyst 1 \
    --repair_multi_gen 1\
    --repair_prompt_num ${repair_prompt_num}\
    --repair_plan 1\
    --output_file_name HumanEvalET_repair_mix_${model}_repair_num_${repair_prompt_num} \
    | tee output_HumanEvalET_repair_mix_${model}_repair_num_${repair_prompt_num}.txt 
    
