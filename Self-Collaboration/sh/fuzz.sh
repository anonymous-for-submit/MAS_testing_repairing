mutate_method=random
python ../main_fuzz_passAt10.py --dataset humaneval \
    --signature \
    --model gpt-3.5-turbo \
    --input_path /data/zlyuaj/muti-agent/fuzzing/output_mutated/original/code_round_0_with_score.jsonl \
    --output_path "your output path" \
    --do_fuzz 0  \
    --only_consider_passed_cases 0 --mutate_method ${mutate_method} \
    --num_generate 10 \
    --alpha 1 \
    --clean_data 0 \
    --num_round 1000 \
    --calc_analyst 1 \
    --mutate_level sentence \
    --split_input 1 \
    --clean_mutate_method 1\
    --parallel 1\
    --save_seed 1 --output_file_name analyst_score_1-1_all_plan_${mutate_method} | tee output_fuzz_pass@10_analyst_score_1-1_all_plan_${mutate_method}.txt 
    # --majority 5 \
    
