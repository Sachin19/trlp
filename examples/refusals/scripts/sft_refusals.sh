mkdir -p huggingface_cache
export HF_HOME="huggingface_cache"
export HF_DATASETS_CACHE="huggingface_cache"

cd /code/examples/refusals/

model_name=$1
epochs=$2

# pip install peft --upgrade
# pip install transformers --upgrade
# pip install bitsandbytes --upgrade
# pip install accelerate --upgrade
# pip install ai2-olmo

datapath=/datasets/
torchrun --nnodes 1  --nproc_per_node 1 scripts/sft.py\
    --model_name=$model_name\
    --streaming\
    --no_gradient_checkpointing\
    --learning_rate 1e-5\
    --num_train_epochs $epochs\
    --per_device_train_batch_size 4\
    --per_device_eval_batch_size 4\
    --data_dir $datapath/safety\
    --data_source jsonl\
    --output_dir /models/\
    --use_lora