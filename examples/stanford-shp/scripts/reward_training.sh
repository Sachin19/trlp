#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --output=./slurm-outputs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50g
#SBATCH -t 0
####SBATCH -x tir-0-9,tir-0-7,tir-0-13,tir-0-15,tir-0-17,tir-0-19,tir-0-11,tir-0-32,tir-0-36,tir-1-13,tir-0-3,tir-1-11,tir-1-28,tir-1-18
#####SBATCH -x tir-0-36,tir-1-28,tir-1-18,tir-1-23,tir-1-13
########SBATCH -x tir-0-19,tir-0-36,tir-0-32,tir-0-17,tir-1-28,tir-1-11,tir-0-11
#### SBATCH -w tir-0-9
#SBATCH -x tir-0-32,tir-0-36,tir-0-11,tir-1-32,tir-1-11,tir-1-23,tir-1-18,tir-1-13,tir-0-3
############,tir-0-7,tir-0-13,tir-0-15,tir-0-17,tir-0-19,tir-0-11,tir-0-32,tir-0-36,tir-1-28,tir-1-18,tir-1-13,tir-0-3

set -x  # echo commands to stdout
set -e  # exit on error
#module load cuda-8.0 cudnn-8.0-5.1
#export CUDE_VISIBLE_DEVICES=2,1
# echo "$@"
# sh "$@"
# echo "ok done"
export HF_DATASETS_CACHE="/projects/tir6/general/sachink/huggingface"
# # conda activate 2022

# export OMP_NUM_THREADS=12
module load cuda-11.1.1 cudnn-11.1.1-v8.0.4.30
module load gcc-7.4
source /projects/tir1/users/sachink/data/anaconda3/bin/activate 2022
# echo "ok done"
# pwd
cd /projects/tir6/general/sachink/personalized-LM/2023/trl/examples/stanford-shp
# accelerate launch scripts/simple-finetuning.py --data_dir $1 --data_prefix $2

# torchrun --nnodes 1  --nproc_per_node 1 
# accelerate launch 
# torchrun --nnodes 1  --nproc_per_node 1  scripts/sft_gpt2.py\
#     --model_name "gpt2-large"\
#     --data_dir /projects/tir6/general/sachink/personalized-LM/2023/steamshp/data\
#     --data_prefix sft_meta-llama-Llama-2-7b-chat-hf_\
#     --instrtype plain\
#     --no_gradient_checkpointing\
#     --learning_rate 1e-5\
#     --seq_length 512\
#     --output_dir /projects/tir6/general/sachink/personalized-LM/2023/models/september/sft/
instrtype=$1
subset=$2
port=$3
torchrun --nnodes 1  --nproc_per_node 1 --rdzv_endpoint 0.0.0.0:$port scripts/reward_modeling_llama.py \
    --model_name "/projects/tir6/general/sachink/personalized-LM/2023/llama/hf_model-7B" \
    --output_dir /projects/tir6/general/sachink/personalized-LM/2023/models/1023-chp/reward_models \
    --data_source chp \
    --data_dir /home/chanyoun/datasets/chp\
    --instrtype $instrtype \
    --subset $subset \


