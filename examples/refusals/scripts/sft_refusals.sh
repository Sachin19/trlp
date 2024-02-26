mkdir -p huggingface_cache
export HF_HOME="huggingface_cache"
export HF_DATASETS_CACHE="huggingface_cache"

cd /code/examples/refusals/

model_name=$1
if [ -z "$model_name" ]
then 
    model_name="meta-llama/Llama-2-7b-hf"
fi

datapath=/datasets/
torchrun --nnodes 1  --nproc_per_node 1 scripts/sft.py\
    --model_name=$model_name\
    --streaming\
    --no_gradient_checkpointing\
    --learning_rate 1e-5\
    --num_train_epochs 3\
    --per_device_train_batch_size 4\
    --per_device_eval_batch_size 4\
    --data_dir $datapath\
    --data_source jsonl\
    --output_dir /models/