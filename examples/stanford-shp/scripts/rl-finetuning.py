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
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

import accelerate

from collections import defaultdict

import logging 
tqdm.pandas()

import pysbd

segmenter = pysbd.Segmenter(language="en", clean=False)
SUBREDDIT2DESCRIPTION = {
    "askculinary" : "/r/AskCulinary provides expert guidance for your specific cooking problems to help people of all skill levels become better cooks, to increase understanding of cooking, and to share valuable culinary knowledge.",
    "askhr" : "A place for employees to ask questions about compensation, benefits, harassment, discrimination, legal, and ethical issues in the workplace.",
    "askdocs" : "Having a medical issue? Ask a doctor or medical professional on Reddit! All flaired medical professionals on this subreddit are verified by the mods.",
    "askanthropology" : "Have you ever wanted to know why humans have been so successful as a species? How societies function without governments, laws, or money? What life was like ten thousand years ago? This is the place to ask!",
    "asksciencefiction" : "**It's like Ask Science, but all questions and answers are written with answers gleaned from the universe itself.** Use in-universe knowledge, rules, and common sense to answer the questions. Or as **fanlore.org** calls it [Watsonian, not a Doylist point of view](http://fanlore.org/wiki/Watsonian_vs._Doylist)",
    "askacademia" : "This subreddit is for discussing academic life, and for asking questions directed towards people involved in academia, (both science and humanities).",
    "askengineers" : "Engineers apply the knowledge of math & science to design and manufacture maintainable systems used to solve specific problems. AskEngineers is a forum for questions about the technologies, standards, and processes used to design & build these systems, as well as for questions about the engineering profession and its many disciplines.",
    "legaladvice" : "A place to ask simple legal questions, and to have legal concepts explained.",
    "explainlikeimfive" : "Explain Like I'm Five is the best forum and archive on the internet for layperson-friendly explanations. Don't Panic!",
    "askbaking" : "Welcome to /r/AskBaking! This subreddit is devoted to the discussion of baking, the questions that arise during the process, and requests for critiques or comments on your work!",
    "askphysics" : "A subreddit to draw simple physics questions away from /r/physics. Ask away.",
    "askscience" : "Ask a science question, get a science answer.",
    "askphilosophy" : "/r/askphilosophy aims to provide serious, well-researched answers to philosophical questions.",
    "askvet" : "A place where you can ask veterinary medicine related questions and get advice from veterinary professionals.",
    "changemyview" : "A place to post an opinion you accept may be flawed, in an effort to understand other perspectives on the issue. Enter with a mindset for conversation, not debate.",
    "askcarguys" : "This is a subreddit for automotive related questions.",
    "askhistorians" : "The Portal for Public History. Please read the rules before participating, as we remove all comments which break the rules. Answers must be in-depth and comprehensive, or they will be removed.",
    "asksocialscience" : "The goal of AskSocialScience is to provide great answers to social science questions, based on solid theory, practice, and research.",
}

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a T5 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config` then run the script with
# `accelerate launch ppo-sentiment-t5-small.py`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="google/flan-t5-large", metadata={"help": "the model name"})
    domain: Optional[str] = field(default="legaladvice", metadata={"help": "the model name, write 'all' for all domains combined"})
    save_model_dir: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/rlhf/", metadata={"help": "the reward model name"})
    reward_model_name: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/reward/flan-t5-large-contextualized", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=2, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

c = 0
lengthcounter = defaultdict(int)

def clean_text(text: str) -> str:
    return text.replace("\n", " ")

def build_shp_dataset(tokenizer, data_dir, input_min_text_length=2, input_max_text_length=8):
    logging.warning("starting data load")
    ds = load_dataset("stanfordnlp/shp", split="train")

    if data_dir != "plain" and data_dir != "contextual" and data_dir != "subredditname":
        ds = ds.filter(lambda x: x["domain"]==f"{data_dir}_train")
    
    logging.warning(f"data size: {len(ds)}")

    cols_to_remove = ds.column_names
    cols_to_remove.remove("history")
    cols_to_remove.remove("domain")
    ds = ds.remove_columns(cols_to_remove)

    logging.warning("Done loading data!")

    def tokenize(sample):
        sample_tokenized = tokenizer.encode(clean_text(sample["history"]))[:400] 
        # sample['query_ids'] = sample_tokenized
        # sample_tokenized = sample_tokenized + tokenizer.encode(". \n\n RESPONSE: ")
        if data_dir == "subredditname":
            domain = sample['domain'].split("_")[0]
            domaindescription = f"SUBREDDIT: {domain} \n\n POST: "
            modelinstruction = f"Write a response to this reddit post in the following subreddit. SUBREDDIT: {domain}. \n\n POST: "
        
        elif data_dir == "contextual":
            domain = sample['domain'].split("_")[0]
            domaindescription = f"SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]} \n\n POST: "
            modelinstruction = f"Write a response to this reddit post in the subreddit with the following description. SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]}. \n\n POST: "

        elif data_dir == "plain":
            domaindescription = "POST: "
            modelinstruction = f"Write a response to this reddit post. \n\n POST: "

        else:
            raise ValueError
        
        sample['query'] = domaindescription + tokenizer.decode(sample_tokenized)
        x = tokenizer.encode(modelinstruction) + sample_tokenized + tokenizer.encode(" \n\n RESPONSE: ")
        lengthcounter[len(x)]+=1
        sample['input_ids'] = x
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds, lengthcounter


def collater(data):
    x =  dict((key, [d[key] for d in data]) for key in data[0])
    return x


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# We retrieve the dataloader by calling the `build_dataset` function.
dataset, lengthcounter = build_shp_dataset(tokenizer, data_dir=script_args.domain)

generation_kwargs = {"top_k": 0.0, "top_p": 1.0, "do_sample": True, "eos_token_id": -1}

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
# ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, optimizer=accelerate.utils.DummyOptim, dataset=dataset, data_collator=collater)
# print(dataset)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collater)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

reward_model_name = script_args.reward_model_name
reward_model = transformers.T5ForConditionalGeneration.from_pretrained(reward_model_name).to(device)
reward_model_tokenizer = transformers.T5Tokenizer.from_pretrained(reward_model_name)

# xlen = len("RESPONSE: ")

modelinstruction = f"Write a response to this reddit post in the following subreddit."    
if script_args.domain == "contextual":
    modelinstruction = f"Write a response to this reddit post in the subreddit with the following description."
elif script_args.domain == "plain":
    modelinstruction = f"Write a response to this reddit post."
instrlen = len(tokenizer.encode(modelinstruction))

def reward_function(query, response, **sent_kwargs):
    # texts = [q[:-xlen] + "RESPONSE A: " + r + "\n\n RESPONSE B: .\n\n Which response is better? RESPONSE" for q, r in zip(query, response)]
    texts = [f"{q} \n\n RESPONSE A: " + r + "\n\n RESPONSE B: .\n\n Which response is better? RESPONSE" for q, r in zip(query, response)]
    # print(texts[0])
    input_ids = reward_model_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").input_ids.to(device)  
    # response_ids = reward_model_tokenizer.encode(texts, padding=False)
    # reward_model_input = reward_model_tokenizer.pad({'input_ids': [(q[instrlen:]+r)[:500] for q, r in zip(input_ids, response_ids)]}, return_tensors="pt").to(device)
    # print(reward_model_input)
    outputs = reward_model.generate(input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    # print(outputs.scores[0])
    # input("dsfds")
    # outputs = model.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    # scores = torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1) # index 71 corresponds to the token for 'A'
    scores = torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1) # index 71 corresponds to the token for 'A'
    # scores = (scores - 0.8) / 0.2
    # input(scores)
    scores = [scores[i] for i in range(scores.size(0))]
    # input(scores)
    # .item()
    #normalize scores
    #TODO

    return scores


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
output_min_length = 100
output_max_length = 200
output_length_sampler = LengthSampler(output_min_length, output_max_length)
reward_model_name = script_args.reward_model_name.split("/")[-1]

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the model
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode([r[1:] for r in response_tensors])
    # batch['query'] = tokenizer.batch_decode(batch["input_ids"])
    
    # Compute reward
    rewards = reward_function(batch['query'], batch["response"], **sent_kwargs)

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # if script_args.save_freq and 
    # script_args.save_freq = 20
    # print(epoch, epoch and epoch % script_args.save_freq == 0)
    # if epoch and epoch % script_args.save_freq == 0:
    if epoch % 100 == 0:
        print(epoch, end="...")
    if epoch == 100 or epoch == 500: # early stop
        print(f"saving to {script_args.save_model_dir}/{reward_model_name}_step_{epoch}")
        model.save_pretrained(f"{script_args.save_model_dir}/{reward_model_name}_step_{epoch}")
        tokenizer.save_pretrained(f"{script_args.save_model_dir}/{reward_model_name}_step_{epoch}")
    

print(f"saving to {script_args.save_model_dir}/{reward_model_name}_step_{epoch}")
model.save_pretrained(f"{script_args.save_model_dir}/{reward_model_name}_step_{epoch}")
tokenizer.save_pretrained(f"{script_args.save_model_dir}/{reward_model_name}_step_{epoch}")
