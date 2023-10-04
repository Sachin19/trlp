####################
#### TODO
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

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

from trl import DPOTrainer

import accelerate

import logging 


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="google/flan-t5-large", metadata={"help": "the model name"})
    model_type: Optional[str] = field(default="plain", metadata={"help": "the model name"})
    save_model_dir: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/dpo", metadata={"help": "the model name"})
    data_dir: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/steamshp/data/dpo", metadata={"help": "the model name, write all for all domains combined"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

post_key = "post_" + script_args.model_type

os.makedirs(script_args.save_model_dir, exist_ok=True)
# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_shp_sft_dataset(tokenizer, data_dir):
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": f"{data_dir}/all_train.json", "valid": f"{data_dir}/all_validation.json"})
    # dataset.set_format("pandas")
    print(dataset)
    max_input_length = 500
    max_target_length = 200

    def preprocess_function(examples):
        features = {}
        features['input_ids'] = tokenizer(
            examples[post_key],
            max_length=max_input_length,
            truncation=True,
            )['input_ids']
        
        text_j = tokenizer(
            examples["text_j"], max_length=max_target_length, truncation=True
        )
        text_k = tokenizer(
            examples["text_k"], max_length=max_target_length, truncation=True
        )
        # print("what", len(text_j.input_ids), len(text_k.input_ids))
        # print([len(item) for item in text_j.input_ids])
        # print([len(item) for item in text_k.input_ids])
        # input()
        features["text_j_input_ids"] = text_j["input_ids"]
        features["text_k_input_ids"] = text_k["input_ids"]
        
        return features



    ds = dataset.map(preprocess_function, batched=True)

    ds.set_format(type="torch")

    return ds


model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name)
ref_model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_shp_sft_dataset(tokenizer, data_dir=script_args.data_dir)


batch_size = script_args.batch_size
num_train_epochs = 1
# Show the training loss with every epoch
logging_steps = len(dataset["train"]) // batch_size
model_name = script_args.model_name.split("/")[-1]


# 2. initialize training arguments:
training_args = TrainingArguments(
    per_device_train_batch_size=script_args.batch_size,
    per_device_eval_batch_size=script_args.batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    remove_unused_columns=False,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    load_best_model_at_end=True,
    output_dir=f"{script_args.save_model_dir}/{script_args.model_type}-{model_name}",
    report_to=script_args.log_with,
    logging_steps=logging_steps,
)


# 3. initialize the DPO trainer
dpo_trainer = DPOTrainer(
    model, ref_model, args=training_args, beta=script_args.beta, train_dataset=dataset["train"], eval_dataset=dataset["valid"], tokenizer=tokenizer
)

# 4. train
dpo_trainer.train()

print("Saving last checkpoint of the model")
model.save_pretrained(f"{script_args.save_model_dir}/{script_args.model_type}-{model_name}_last_checkpoint")