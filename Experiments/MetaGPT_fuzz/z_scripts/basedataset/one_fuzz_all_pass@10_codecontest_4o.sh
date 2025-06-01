
model=gpt-4o
name_tag=basedataset
dataset=codecontest
num_generate=10
run_generate=1
run_evaluate=1
python /home/zlyuaj/muti-agent/MetaGPT/metagpt/software_main_fuzzing_BaseDataset.py  \
    --model ${model}\
    --input_path /home/zlyuaj/muti-agent/MetaGPT/data/CodeContest_Test.jsonl \
    --output_path /home/zlyuaj/muti-agent/MetaGPT/output/${name_tag}/ \
    --dataset ${dataset}\
    --output_file_name ${dataset}_${model} \
    --workspace workspace_${model}_${dataset}_${name_tag} \
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --parallel 1\
    | tee output_${model}_${dataset}_${name_tag}.txt  
    # --majority 5 \
    
