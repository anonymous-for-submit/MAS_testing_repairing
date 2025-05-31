mutate_method=random
model=gpt-35-turbo
run_multi_gen=1
repair_prompt_num=2
repair_plan=1
add_monitor=1
repair_code=1
python ../main_fuzz_passAt10.py --dataset humaneval \
    --signature \
    --model ${model} \
    --run_multi_gen ${run_multi_gen}\
    --repair_prompt_num ${repair_prompt_num}\
    --repair_plan ${repair_plan}\
    --add_monitor ${add_monitor}\
    --repair_code ${repair_code}\
    --input_path  \
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
    --save_seed 1 --output_file_name repair_${model}_analyst_score_1-1 | tee repair_${model}_analyst_score_1-1_all_plan_${mutate_method}.txt 
    # --majority 5 \
    
