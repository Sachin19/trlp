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
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,DataCollatorForSeq2Seq

import accelerate

import logging 
tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="google/flan-t5-large", metadata={"help": "the model name"})
    save_model_dir: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/sft", metadata={"help": "the model name"})
    data_dir: Optional[str] = field(default="data", metadata={"help": "the model name, write all for all domains combined"})
    data_prefix: Optional[str] = field(default="sft_", metadata={"help": "the model name, write all for all domains combined"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

os.makedirs(script_args.save_model_dir, exist_ok=True)
# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_shp_sft_dataset(tokenizer, data_dir, data_prefix="sft_"):
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": f"{data_dir}/{data_prefix}train.json", "valid": f"{data_dir}/{data_prefix}validation.json"})
    # dataset.set_format("pandas")
    print(dataset)
    max_input_length = 400
    max_target_length = 200

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["x"],
            max_length=max_input_length,
            truncation=True,
            )
        labels = tokenizer(
            examples["y"], max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = dataset.map(preprocess_function, batched=True)
    ds.set_format(type="torch")

    return ds


model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_shp_sft_dataset(tokenizer, data_dir=script_args.data_dir, data_prefix=script_args.data_prefix)


batch_size = 1
num_train_epochs = 1
# Show the training loss with every epoch
logging_steps = len(dataset["train"]) // batch_size
model_name = script_args.model_name.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{script_args.save_model_dir}/{script_args.data_prefix}-{model_name}",
    evaluation_strategy="epoch",
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.batch_size,
    per_device_eval_batch_size=script_args.batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=num_train_epochs,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    predict_with_generate=False,
    logging_steps=logging_steps,
    push_to_hub=False,
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# dataset = dataset.remove_columns(
    # books_dataset["train"].column_names
# )

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.evaluate()