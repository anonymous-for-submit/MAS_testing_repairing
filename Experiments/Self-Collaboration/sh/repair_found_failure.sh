model=gpt-35-turbo
run_multi_gen=1
repair_prompt_num=2
repair_plan=1
add_monitor=1
repair_code=1
python /data/zlyuaj/muti-agent/fuzzing/main_repair.py \
    --model ${model} \
    --run_multi_gen ${run_multi_gen}\
    --repair_prompt_num ${repair_prompt_num}\
    --repair_plan ${repair_plan}\
    --add_monitor ${add_monitor}\
    --repair_code ${repair_code}\
    --data_path /data/zlyuaj/muti-agent/fuzzing/output_fuzzing_one_per_time/pass@10/split_input/sentence_level/alabation_mutate_method/ \
    --num_generate 10 | tee found_failure_${model}_repair_prompt_num_${repair_prompt_num}_repair_plan_repair_code.txt 
    
    
