#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --output=./slurm-outputs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50g
#SBATCH -t 0
#SBATCH -x tir-0-32,tir-0-36,tir-0-11,tir-1-32,tir-1-11,tir-1-23,tir-0-9,tir-0-17,tir-1-18,tir-0-3

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
# pwd
cd /projects/tir6/general/sachink/personalized-LM/2023/trl/examples/stanford-shp
nvidia-smi
# # #############NO TRAINING
# python evaluation/generate_responses.py --model_name google/flan-t5-large --output_file outputs/top_p/flan-t5-large-instr-contextualized.txt --instrtype contextualized
# python evaluation/generate_responses.py --model_name google/flan-t5-large --output_file outputs/top_p/flan-t5-large-instr-subredditname.txt --instrtype subredditname
# python evaluation/generate_responses.py --model_name google/flan-t5-large --output_file outputs/top_p/flan-t5-large-instr-plain.txt --instrtype plain

# ############SFT
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/sft/sft_contextualized-flan-t5-large/checkpoint-2500 --output_file outputs/top_p/sft_flan-t5-large-contextualized.txt --instrtype sft_contextualized
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/sft/sft_subredditname_-flan-t5-large/checkpoint-2500 --output_file outputs/top_p/sft_flan-t5-large-subredditname.txt --instrtype sft_subredditname
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/sft/sft_plain_-flan-t5-large/checkpoint-2500 --output_file outputs/top_p/sft_flan-t5-large-plain.txt --instrtype sft_plain

#########DPO
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/dpo/contextualized-flan-t5-large_last_checkpoint --output_file outputs/top_p/dpo_flan-t5-large-contextualized.txt --instrtype contextualized
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/dpo/plain-flan-t5-large_last_checkpoint --output_file outputs/top_p/dpo_flan-t5-large-plain.txt --instrtype plain
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/dpo/subredditname-flan-t5-large_last_checkpoint --output_file outputs/top_p/dpo_flan-t5-large-subredditname.txt --instrtype subredditname

# # #########RL
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf-new/contextualized_deberta-v3-large_steamshp_2e-05_last_checkpoint_step_100 --output_file outputs/top_p/rlhf_new_flan-t5-large-contextualized-step-100.txt --instrtype contextualized
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf-new/contextualized_deberta-v3-large_steamshp_2e-05_last_checkpoint_step_500 --output_file outputs/top_p/rlhf_new_flan-t5-large-contextualized-step-500.txt --instrtype contextualized

# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf-new/plain_deberta-v3-large_steamshp_2e-05_last_checkpoint_step_100 --output_file outputs/top_p/rlhf_new_flan-t5-large-plain-step-100.txt --instrtype plain
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf-new/plain_deberta-v3-large_steamshp_2e-05_last_checkpoint_step_500 --output_file outputs/top_p/rlhf_new_flan-t5-large-plain-step-500.txt --instrtype plain

# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf-new/subredditname_deberta-v3-large_steamshp_2e-05_last_checkpoint_step_100 --output_file outputs/top_p/rlhf_new_flan-t5-large-subredditname-step-100.txt --instrtype subredditname
# python evaluation/generate_responses.py --model_name /projects/tir6/general/sachink/personalized-LM/2023/models/rlhf-new/subredditname_deberta-v3-large_steamshp_2e-05_last_checkpoint_step_500 --output_file outputs/top_p/rlhf_new_flan-t5-large-subredditname-step-500.txt --instrtype subredditname

# ############
algorithm=$1
bs=$2
mkdir -p outputs/0923-newreddit/
python evaluation/inference_llama.py --output_dir outputs/0923-newreddit/ --algorithm ${algorithm} --instrtype subredditname --subset explainlikeimfive --batch_size $bs
python evaluation/inference_llama.py --output_dir outputs/0923-newreddit/ --algorithm ${algorithm} --instrtype subredditname --subset all --batch_size $bs
python evaluation/inference_llama.py --output_dir outputs/0923-newreddit/ --algorithm ${algorithm} --instrtype plain --subset all --batch_size $bs
python evaluation/inference_llama.py --output_dir outputs/0923-newreddit/ --algorithm ${algorithm} --instrtype contextualized --subset all --batch_size $bs
python evaluation/inference_llama.py --output_dir outputs/0923-newreddit/ --algorithm ${algorithm} --instrtype subredditname --subset askphysics --batch_size $bs