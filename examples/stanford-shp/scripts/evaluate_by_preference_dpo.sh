#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --output=./slurm-outputs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=25g
#SBATCH -t 0
####SBATCH -x tir-0-9,tir-0-7,tir-0-13,tir-0-15,tir-0-17,tir-0-19,tir-0-11,tir-0-32,tir-0-36,tir-1-13,tir-0-3,tir-1-11,tir-1-28,tir-1-18
#####SBATCH -x tir-0-36,tir-1-28,tir-1-18,tir-1-23,tir-1-13
########SBATCH -x tir-0-19,tir-0-36,tir-0-32,tir-0-17,tir-1-28,tir-1-11,tir-0-11
#SBATCH -x tir-0-9,tir-0-7,tir-0-13,tir-0-15,tir-0-17,tir-0-19,tir-0-11,tir-0-32,tir-0-36,tir-1-28,tir-1-18,tir-1-13,tir-0-3

set -x  # echo commands to stdout
set -e  # exit on error
#module load cuda-8.0 cudnn-8.0-5.1
#export CUDE_VISIBLE_DEVICES=2,1
# echo "$@"
# sh "$@"
# echo "ok done"
export HF_DATASETS_CACHE="/projects/tir6/general/sachink/huggingface"
# conda activate 2022

export OMP_NUM_THREADS=12
module load cuda-11.1.1 cudnn-11.1.1-v8.0.4.30
module load gcc-7.4
source /projects/tir1/users/sachink/data/anaconda3/bin/activate 2022
# echo "ok done"
pwd
cd /projects/tir6/general/sachink/personalized-LM/2023/trl/examples/stanford-shp

# declate -a paths1=()
# declare -a paths2=("flan-t5-large-instr-contextualized.txt" "flan-t5-large-instr-plain.txt" "flan-t5-large-instr-subredditname.txt" "/projects/tir6/general/sachink/personalized-LM/2023/models/sft/sft_contextualized-flan-t5-large" "/projects/tir6/general/sachink/personalized-LM/2023/models/sft/sft_plain-flan-t5-large" "/projects/tir6/general/sachink/personalized-LM/2023/models/sft/sft_subredditname-flan-t5-large")

# # Iterate the string array using for loop
# for val in ${paths[@]}; do
#  echo $val
# done
reward_model_name=/projects/tir6/general/sachink/personalized-LM/2023/models/reward/contextualized_flan-t5-large
#pretrained

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/flan-t5-large-instr-contextualized.txt

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/flan-t5-large-instr-subredditname.txt

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/flan-t5-large-instr-plain.txt

#simple finetuned
python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/sft_flan-t5-large-contextualized.txt

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/sft_flan-t5-large-subredditname.txt

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/sft_flan-t5-large-plain.txt

#rl-tuned 100
python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/rlhf_new_flan-t5-large-contextualized-step-100.txt

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/rlhf_new_flan-t5-large-plain-step-100.txt

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/rlhf_new_flan-t5-large-subredditname-step-100.txt

#rl-tuned 500
python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/rlhf_new_flan-t5-large-contextualized-step-500.txt

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/rlhf_new_flan-t5-large-plain-step-500.txt

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/rlhf_new_flan-t5-large-subredditname-step-500.txt

# other dpo
python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/dpo_flan-t5-large-plain.txt

python evaluation/evaluate_by_preference.py\
    --reward_model_name $reward_model_name\
    --generations_file1 outputs/top_p/dpo_flan-t5-large-contextualized.txt\
    --generations_file2 outputs/top_p/dpo_flan-t5-large-subredditname.txt


# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf/contextualized_flan-t5-large_step_100 --output_file outputs/top_p/rlhf_flan-t5-large-contextualized-step-100.txt --instrtype contextualized
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf/contextualized_flan-t5-large_step_500 --output_file outputs/top_p/rlhf_flan-t5-large-contextualized-step-500.txt --instrtype contextualized

# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf/subredditname_flan-t5-large_step_100 --output_file outputs/top_p/rlhf_flan-t5-large-subredditname-step-100.txt --instrtype subredditname
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf/subredditname_flan-t5-large_step_500 --output_file outputs/top_p/rlhf_flan-t5-large-subredditname-step-500.txt --instrtype subredditname

# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf/plain_flan-t5-large_step_100 --output_file outputs/top_p/rlhf_flan-t5-large-plain-step-100.txt --instrtype plain
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf/plain_flan-t5-large_step_500 --output_file outputs/top_p/rlhf_flan-t5-large-plain-step-500.txt --instrtype plain
