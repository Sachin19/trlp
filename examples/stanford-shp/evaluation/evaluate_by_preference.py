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
from transformers import AutoTokenizer, HfArgumentParser, T5ForConditionalGeneration

import sys
import time

import logging

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

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    reward_model_name: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-legaladvice_rl_step_25", metadata={"help": "the reward model name"})
    # reward_model_name: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-legaladvice_rl_step_25", metadata={"help": "the reward model name"})
    #input_file: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-legaladvice_rl_step_25", metadata={"help": "the reward model name"})
    generations_file1: Optional[str] = field(default=None, metadata={"help": ""})
    generations_file2: Optional[str] = field(default=None, metadata={"help": ""})
    instr_type: Optional[str] = field(default=None, metadata={"help": ""})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

batchsize = 64


print("Loading dataset...")
ds = load_dataset("stanfordnlp/shp", split="test")
# if data_dir != "all" and data_dir != "contextual":
# ds = ds.filter(lambda x: x["domain"]==f"{data_dir}_train")
print(len(ds))
cols_to_remove = ds.column_names
cols_to_remove.remove("history")
cols_to_remove.remove("post_id")
cols_to_remove.remove("domain")
# print(len(ds), cols_to_remove)
ds = ds.remove_columns(cols_to_remove)
df = ds.to_pandas()

coveredposts = set()

logging.warning("Done loading data!")

def clean_text(text: str) -> str:
    return text.replace("\n", " ")

import json
for top_p in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    text_batch = []
    fin1 = open(script_args.generations_file1)
    fin2 = open(script_args.generations_file2)
    # xlen = len("RESPONSE: ")
    alltexts = []
    for i, line in enumerate(fin1):
        line2 = fin2.readline()
        items = json.loads(line)
        items2 = json.loads(line2)
        domain = df.iloc[i]['domain'].split("_")[0]
        # print(domain)
        post = items["post"]
        response = items["response"][str(top_p)]
        response2 = items2["response"][str(top_p)]
        # print(items, items2)
        # input()
        if script_args.instr_type == "subredditname":
            alltexts.append(f"SUBREDDIT: {domain}\n\n POST: {post} \n\n RESPONSE A: {response}\n\n RESPONSE B: {response2}.\n\n Which response is better? RESPONSE")
        elif script_args.instr_type == "subredditdesc":
            alltexts.append(f"SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]}\n\n POST: {post} \n\n RESPONSE A: {response}\n\n RESPONSE B: {response2}.\n\n Which response is better? RESPONSE")
        else:
            alltexts.append(f"POST: {post} \n\n RESPONSE A: {response}\n\n RESPONSE B: {response2}.\n\n Which response is better? RESPONSE")
        
        
    reward_model_name = script_args.reward_model_name
    reward_model = transformers.T5ForConditionalGeneration.from_pretrained(reward_model_name).to("cuda")
    reward_model_tokenizer = transformers.T5Tokenizer.from_pretrained(reward_model_name)

    acount = 0
    total = 0
    from tqdm import tqdm

    for i in tqdm(range(0, len(alltexts), batchsize)):
        text_batch = alltexts[i:min(len(alltexts), i + batchsize)]
        
        st = time.time()
        tokens = reward_model_tokenizer.prepare_seq2seq_batch(text_batch, return_tensors="pt", max_length=1000, truncation=True) # check
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
            torch.cuda.empty_cache()
        # print(time.time() - st)
        st=time.time()
        
        # input_ids = reward_model_tokenizer(tokens, padding=True, truncation=True, max_length=1000, return_tensors="pt").input_ids.to("cuda")  
        outputs = reward_model.generate(**tokens, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        
        favorA = outputs.scores[0][:, 71] > outputs.scores[0][:, 272]
        # print(scores)
        # print(favorA)
        # input()
        torch.cuda.empty_cache()
        del tokens
        del outputs

        acount += favorA.int().sum()
        total += favorA.size(0)
        # fout.write("\n".join(favorA) + "\n")
        # fout.flush()

print(f"A is favored over B {acount} out of {total} or {acount/total}% of the times")

# fout.close()

