# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional
import os

import torch
from datasets import load_dataset
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, TrainingArguments

import accelerate

import logging 

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
tqdm.pandas()


@dataclass
class ScriptArguments:
    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    #try gpt2, gpt2-xl, gpt-neo models, pythia maybe, llama2 in the other file, only need to modify prepare_sample_text function and the rest should just work fine
    # also need to format for sft per subreddit, add subreddit info in the log itself?
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    output_dir: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/sft", metadata={"help": "the model name"})
    data_dir: Optional[str] = field(default="data", metadata={"help": "the model name, write all for all domains combined"})
    data_prefix: Optional[str] = field(default="sft_", metadata={"help": "the model name, write all for all domains combined"})
    subset: Optional[str] = field(default="all", metadata={"help": "the model name, write all for all domains combined"})
    instrtype: Optional[str] = field(default="sft_", metadata={"help": "the model name, write all for all domains combined"})
    
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=0, metadata={"help": "the number of workers"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})

    max_steps: Optional[int] = field(default=500, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=10, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

os.makedirs(script_args.output_dir, exist_ok=True)
# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.

def prepare_sample_text_plain(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    text = example["input_plain"] + example["output"]
    return text

def prepare_sample_text_contextualized(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    text = example["input_contextualized"] + example["output"]
    return text

def prepare_sample_text_subredditname(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    text = example["input_subredditname"] + example["output"]
    return text

prepare_sample_text_fns ={
    "plain": prepare_sample_text_plain,
    "contextualized": prepare_sample_text_contextualized,
    "subredditname": prepare_sample_text_subredditname,
}


def chars_token_ratio(dataset, tokenizer, instrtype, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text_fns[instrtype](example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    print(total_characters, total_tokens)
    return total_characters / total_tokens


def create_datasets(tokenizer, args):
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_dir = args.data_dir
    data_prefix = args.data_prefix
    instrtype = args.instrtype
    
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": f"{data_dir}/{data_prefix}train.json", "valid": f"{data_dir}/{data_prefix}validation.json"})
    # dataset.set_format("pandas")

    if args.subset == "all":
        train_data = dataset["train"]
        print(train_data)
        valid_data = dataset["valid"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    else:
        train_data = dataset["train"].filter(lambda example: example['domain'] == args.subset+"_train")
        print(train_data)
        valid_data = dataset["valid"].filter(lambda example: example['domain'] == args.subset+"_validation")
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    # if args.streaming:
    #     print("Loading the dataset in streaming mode")
    #     valid_data = dataset.take(args.size_valid_set)
    #     train_data = dataset.skip(args.size_valid_set)
    #     train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    # else:
        # dataset = dataset.train_test_split(test_size=0.005, seed=None)
        # train_data = dataset["train"]
        # valid_data = dataset["test"]
        # print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, instrtype)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text_fns[instrtype],
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text_fns[instrtype],
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


base_model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

# We retrieve the dataloader by calling the `build_dataset` function.
train_dataset, eval_dataset = create_datasets(tokenizer, script_args)


batch_size = 1
num_train_epochs = 1
# Show the training loss with every epoch
# logging_steps = len(dataset["train"]) // batch_size
model_name = script_args.model_name.split("/")[-1]

print(base_model.device)
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    save_total_limit=2,
    remove_unused_columns=False,
    run_name="sft_gpt2",
    bf16=True,
    report_to=script_args.log_with,
)
print(train_dataset)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
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
# del base_model
# torch.cuda.empty_cache()

# model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
# model = model.merge_and_unload()

# output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
# model.save_pretrained(output_merged_dir, safe_serialization=True)


############
# args = Seq2SeqTrainingArguments(
#     output_dir=f"{script_args.save_model_dir}/{script_args.data_prefix}-{model_name}",
#     evaluation_strategy="epoch",
#     learning_rate=script_args.learning_rate,
#     per_device_train_batch_size=script_args.batch_size,
#     per_device_eval_batch_size=script_args.batch_size,
#     weight_decay=0.01,
#     save_total_limit=2,
#     num_train_epochs=num_train_epochs,
#     gradient_accumulation_steps=script_args.gradient_accumulation_steps,
#     predict_with_generate=False,
#     logging_steps=logging_steps,
#     push_to_hub=False,
# )


# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# # dataset = dataset.remove_columns(
#     # books_dataset["train"].column_names
# # )

# from transformers import Seq2SeqTrainer

# trainer = Seq2SeqTrainer(
#     model,
#     args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["valid"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )

# trainer.train()
# trainer.evaluate()