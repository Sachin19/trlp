# Fine-Tune causal LMs on reddit or stackoverflow datasets
import os
import json 

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

from functools import partial

from constants import TOKEN

@dataclass
class ScriptArguments:
    data_source: Optional[str] = field(default="shp", metadata={"help": "the model name, write all for all domains combined"})
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    output_dir: Optional[str] = field(default="trained_models/shp/sft", metadata={"help": "the model name"})
    data_dir: Optional[str] = field(default="data", metadata={"help": "the model name, write all for all domains combined"})
    train_data_path: Optional[str] = field(default=None, metadata={"help": "the model name, write all for all domains combined"})
    test_data_path: Optional[str] = field(default=None, metadata={"help": "the model name, write all for all domains combined"})
    data_prefix: Optional[str] = field(default="sft_", metadata={"help": "the model name, write all for all domains combined"})

    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})

    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    max_steps: Optional[int] = field(default=500, metadata={"help": "the maximum number of sgd steps"})
    num_train_epochs: Optional[int] = field(default=5, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=64, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=True, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=False, metadata={"help": "whether to use packing for SFTTrainer"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    use_4bit: Optional[bool] = field(default=False, metadata={"help": ""})
    use_lora: Optional[bool] = field(default=False, metadata={"help": ""})


    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    # num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "warmup ratio"})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    #output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    score_ratio_threshold: Optional[float] = field(default=2.0)
    num_examples_per_post: Optional[float] = field(default=5)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
script_args.output_dir = os.path.join(script_args.output_dir, "sft", script_args.model_name.replace("/", "-"))
os.makedirs(script_args.output_dir, exist_ok=True)

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def preprocess_function(example):
    """Prepare the text from a sample of the dataset."""
    text = ""
    for message in example['messages']:
        if message["role"] == "system":
            text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    # text += f"{tokenizer.eos_token}"

    return text


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = preprocess_function(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    print(total_characters, total_tokens)
    return total_characters / (total_tokens+1e-6)


def create_datasets_from_jsonl(tokenizer, args, load_only=["messages"]):
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    data_path = args.data_dir # will contain train, validation
    # features = None
    if args.train_data_path is not None:
        train_path = args.train_data_path
        test_path = args.test_data_path
    else:
        train_path = args.data_path + "_train.jsonl"
        test_path = args.data_path + "_test.jsonl"

    if len(load_only) > 0:
        def get_datadict(filepath):
            with open(filepath) as fin:
                data = [json.loads(line) for line in fin]
                data = [{feature: item[feature] for feature in load_only} for item in data]
                return {key: [dic[key] for dic in data] for key in data[0]}

        train_dict = get_datadict(train_path)
        val_dict = get_datadict(test_path)
        train_dataset = Dataset.from_dict(train_dict)
        validation_dataset = Dataset.from_dict(val_dict)
        dataset = DatasetDict({'train': train_dataset, 'validation': validation_dataset})
        # from datasets import Features, Value
        # feature_dict = {feature: Value('string') for feature in load_only}
        # features = Features(feature_dict)
    else:
        dataset = load_dataset("json", data_files={
            "train": train_path,
            "validation": test_path
            }, features=features)
    
    print(dataset)
    train_data = dataset["train"]
    valid_data = dataset["validation"]

    print(f"Training data size: {len(train_data)}")
    print(f"Held-out data size: {len(valid_data)}")
    
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    
    train_dataset = train_data.map(lambda x: {'text': preprocess_function(x)})
    valid_dataset = valid_data.map(lambda x: {'text': preprocess_function(x)})

    return train_dataset, valid_dataset

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, token=TOKEN)

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  

if script_args.data_source == "jsonl": 
    train_dataset, eval_dataset = create_datasets_from_jsonl(tokenizer, script_args)
else:
    raise ValueError("expecting a dataset in jsonl format")

print("Datasets loaded")

bnb_config = None
if script_args.use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

if script_args.use_4bit:
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        # use_auth_token=True,
        token=TOKEN
    )
else: 
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        device_map={"": 0},
        trust_remote_code=True,
        # use_auth_token=True,
        token=TOKEN
    )
    
base_model.config.use_cache = False

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    # max_steps=script_args.max_steps,
    num_train_epochs=script_args.num_train_epochs,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    eval_steps=script_args.eval_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    # warmup_steps=script_args.num_warmup_steps,
    warmup_ratio=script_args.warmup_ratio,
    optim=script_args.optimizer_type,
    bf16=True,
    remove_unused_columns=False,
    run_name="sft_refusals",
    ddp_find_unused_parameters=False,
)

if script_args.use_lora:
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        packing=script_args.packing,
        max_seq_length=script_args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
else:
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        packing=script_args.packing,
        max_seq_length=script_args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
trainer.train()
trainer.save_model(script_args.output_dir)

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
