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

import torch
from datasets import load_dataset

import transformers
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, AutoModelForCausalLM


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-legaladvice_rl_step_25", metadata={"help": "the reward model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

generation_kwargs = {"top_k": 0.0, "top_p": 0.9, "do_sample": True, "eos_token_id": -1}


device=0
if not torch.cuda.is_available():
    device="cpu"  

model.half().to(device)

while True:
    no_of_lines = 1
    lines = ""
    print(f"Enter the prompt ({no_of_lines} lines): ")
    for i in range(no_of_lines):
        lines += input()+"\n\n"
    n = int(input("Output length: " ))

    # input(batch)
    query_tensors = tokenizer.encode("Write a response to this reddit post. \n\n "+lines, return_tensors="pt").to(device)
    print("Output: ", end="")
    # Get response from t5
    response_tensor = model.generate(query_tensors, max_new_tokens=n, **generation_kwargs)
    response = tokenizer.decode(response_tensor[0], skip_special_tokens=True)

    print(response)