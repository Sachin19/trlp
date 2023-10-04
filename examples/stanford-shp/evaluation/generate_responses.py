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

from tqdm import tqdm
import logging

import json

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
    model_name: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-legaladvice_rl_step_25", metadata={"help": "the reward model name"})
    tokenizer_name: Optional[str] = field(default="google/flan-t5-large", metadata={"help": "the model tokenizer"})
    #input_file: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-legaladvice_rl_step_25", metadata={"help": "the reward model name"})
    output_file: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    instrtype: Optional[str] = field(default=None, metadata={"help": "the reward model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print("loading model and tokenizer")
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
model = T5ForConditionalGeneration.from_pretrained(script_args.model_name).cuda().half()
print("model loaded")
model = model.eval()

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
# data_module = dict(
#     train_dataset=data['train'],
#     eval_dataset=data['validation'],
#     data_collator=DataCollatorForData2TextLanguageModeling(tokenizer=tokenizer),
# )
c = 0

def clean_text(text: str) -> str:
    return text.replace("\n", " ")

print("read the file with", ds.shape[0], "entries")
batchsize = 64
x = 0
st=time.time()
emptylines = []
text_batch = []
posttext_batch = []

fout = open(script_args.output_file, "w")
# reward_model_name = script_args.reward_model_name
# reward_model = transformers.T5ForConditionalGeneration.from_pretrained(reward_model_name).to(device)
# reward_model_tokenizer = transformers.T5Tokenizer.from_pretrained(reward_model_name)

with torch.no_grad():
    for i, row in tqdm(df.iterrows(), total=df.shape[0]): 
    # for i in range(0, len(ds), batchsize):
        # print(i, coveredposts)
        if i == 0 or len(text_batch) < batchsize:
            # print(i, len(text_batch), batchsize)
            if row['post_id'] not in coveredposts:
                cleanpost = clean_text(row["history"])
                posttext_batch.append(cleanpost)
                if script_args.instrtype == "sft_plain":
                    text_batch.append(f"POST: {cleanpost}. \n\n What is a good response to this post? RESPONSE: ")
                elif script_args.instrtype == "sft_contextualized":
                    domain = row['domain'].split("_")[0]
                    text_batch.append(f"SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]} \n\n POST: {cleanpost}\n\n What is a good response to this post? RESPONSE: ")
                elif script_args.instrtype == "sft_subredditname":
                    domain = row['domain'].split("_")[0]
                    text_batch.append(f"SUBREDDIT: {domain} \n\n POST: {cleanpost}\n\n What is a good response to this post? RESPONSE: ")
                elif script_args.instrtype == "plain":
                    text_batch.append(f"Write a response to this reddit post. \n\n POST: {cleanpost} \n\n RESPONSE: ")
                elif script_args.instrtype == "contextualized":
                    domain = row['domain'].split("_")[0]
                    text_batch.append(f"Write a response to this reddit post in the subreddit with the following description. SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]} \n\n POST: {cleanpost} \n\n RESPONSE: ")
                elif script_args.instrtype == "subredditname":
                    domain = row['domain'].split("_")[0]
                    text_batch.append(f"Write a response to this reddit post in the following subreddit. SUBREDDIT: {domain}. \n\n POST: {cleanpost} \n\n RESPONSE: ")
                else:
                    raise ValueError
                    
                coveredposts.add(row['post_id'])
            continue
        
        # print(i, len(text_batch), flush=True)
        # input()
        st = time.time()
        # print(text_batch)
        tokens = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True) # check
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
            torch.cuda.empty_cache()
        # print(time.time() - st)
        st=time.time()
        text_outputs = [{} for _ in range(len(text_batch))]
        for top_p in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            generation_kwargs = {"top_p": top_p, "do_sample": True}
            outputs = model.generate(**tokens, max_new_tokens=200, **generation_kwargs) # check
            for i, output in enumerate(outputs):
                text_outputs[i][top_p] = tokenizer.decode(output, skip_special_tokens=True)
            # print(len(text_outputs), flush=True)
            # input()
            # print(time.time()-st)
            # del tokens
            # del outputs
        filewrite = [json.dumps({"post": post, "response": response}) for post, response in zip(posttext_batch, text_outputs)]

        # print(filewrite)
        # input()
        fout.write("\n".join(filewrite) + "\n")
        fout.flush()

        x += len(text_outputs)
        text_batch = []
        posttext_batch = []

        if i % 50 == 0:
            print(f"done {x} sentences, took {(time.time()-st)/(x)} seconds per sentence", flush=True)

print(f"wrote {x} lines")

fout.close()

