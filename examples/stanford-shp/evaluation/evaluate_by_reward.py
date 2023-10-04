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
    generations_file: Optional[str] = field(default=None, metadata={"help": "the reward model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

batchsize = 64
text_batch = []

fin = open(generations_file)
# xlen = len("RESPONSE: ")
alltexts = []
for line in fin:
    items = json.read(line)
    post = items["post"]
    response = items["response"]
    alltexts.append("POST: " + post + " \n\n RESPONSE A: " + response + "\n\n RESPONSE B: .\n\n Which response is better? RESPONSE")
    
    
reward_model_name = script_args.reward_model_name
reward_model = transformers.T5ForConditionalGeneration.from_pretrained(reward_model_name).to(device)
reward_model_tokenizer = transformers.T5Tokenizer.from_pretrained(reward_model_name)


for i in range(0, len(alltexts), batchsize):
    text_batch = alltexts[i:min(len(alltexts), i + batchsize)]
    
    st = time.time()
    tokens = tokenizer.prepare_seq2seq_batch(text_batch, return_tensors="pt", max_length=1000, truncate=True) # check
    for k in tokens.keys():
        tokens[k] = tokens[k].cuda()
        torch.cuda.empty_cache()
    # print(time.time() - st)
    st=time.time()
    
    input_ids = reward_model_tokenizer(texts, padding=True, truncation=True, max_length=1000, return_tensors="pt").input_ids.to(device)  
    outputs = reward_model.generate(input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    
    scores = torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1) # index 71 corresponds to the token for 'A'
    scores = [str(scores[i]) for i in range(scores.size(0))]
    print(scores)
    input()
    torch.cuda.empty_cache()
    del tokens
    del outputs

    fout.write("\n".join(scores) + "\n")
    fout.flush()

fout.close()

