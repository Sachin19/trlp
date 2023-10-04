# %%
from typing import Optional

import torch
from datasets import load_dataset

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM

from peft import AutoPeftModelForCausalLM, PeftModel

from dataclasses import dataclass, field
from typing import Optional

import logging
import pysbd

from tqdm import tqdm

import time
import os
import json
# %%
SUBREDDITS = [
    "askculinary",
    "askhr",
    "askdocs",
    "askanthropology",
    "asksciencefiction",
    "askacademia",
    "askengineers",
    "legaladvice",
    "explainlikeimfive",
    "askbaking",
    "askphysics",
    "askscience",
    "askphilosophy",
    "askvet",
    "changemyview",
    "askcarguys",
    "askhistorians",
    "asksocialscience",
]

SUBREDDITS_NORMVIO = ["askphilosophy", "changemyview", "explainlikeimfive", "legaladvice"]

ANTHROPIC = [
    "anthropic_helpful",
    "anthropic_harmful",
][:1]

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
    tokenizer_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model tokenizer"})
    #input_file: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-legaladvice_rl_step_25", metadata={"help": "the reward model name"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    instrtype: Optional[str] = field(default="plain", metadata={"help": "the reward model name"})
    subset: Optional[str] = field(default="all", metadata={"help": "the reward model name"})
    algorithm: Optional[str] = field(default="sft", metadata={"help": "the reward model name"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "decoding batch size"})
    single_peft_load: Optional[bool] = field(default=False, metadata={"help": "load main model and add peft to it after"})



parser =     HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# %%
########## load finetuned models
# model_name = "../../../../llama/hf_model-7B/"
#model_name = "meta-llama/Llama-2-7b-chat-hf"
device=0
if not torch.cuda.is_available():
    device="cpu"  

model_names = {"plain_all": {}, "subredditname_all": {}, "subredditname_askphysics": {}, "subredditname_explainlikeimfive": {}, "contextualized_all": {}}

# plain_all
model_names["plain_all"]["sft"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/sft/llama-7B_plain_all/final_checkpoint"
model_names["plain_all"]["dpo"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/dpo/llama-7B_plain_all/"
model_names["plain_all"]["ppo"] = "/projects/tir5/users/sachink/personalized-LM/0923-newreddit/rlhf/llama-se-rl-finetune-128-8-8-1.4e-5_adam_plain_allstep_400"
# subredditname_all
model_names["subredditname_all"]["sft"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/sft/llama-7B_subredditname_all/final_checkpoint"
model_names["subredditname_all"]["dpo"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/dpo/llama-7B_subredditname_all/"
model_names["subredditname_all"]["ppo"] = "/projects/tir5/users/sachink/personalized-LM/0923-newreddit/rlhf/llama-se-rl-finetune-128-8-8-1.4e-5_adam_subredditname_allstep_400"
# subredditname_askphysics
model_names["subredditname_askphysics"]["sft"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/sft/llama-7B_subredditname_askphysics/final_checkpoint"
model_names["subredditname_askphysics"]["dpo"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/dpo/llama-7B_subredditname_askphysics/"
model_names["subredditname_askphysics"]["ppo"] = "/projects/tir5/users/sachink/personalized-LM/0923-newreddit/rlhf/llama-se-rl-finetune-128-8-8-1.4e-5_adam_subredditname_askphysicsstep_200"
# subredditname_explainlikeimfive 
model_names["subredditname_explainlikeimfive"]["sft"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/sft/llama-7B_subredditname_explainlikeimfive/final_checkpoint"
model_names["subredditname_explainlikeimfive"]["dpo"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/dpo/llama-7B_subredditname_explainlikeimfive/"
model_names["subredditname_explainlikeimfive"]["ppo"] = "/projects/tir5/users/sachink/personalized-LM/0923-newreddit/rlhf/llama-se-rl-finetune-128-8-8-1.4e-5_adam_subredditname_explainlikeimfivestep_200"
# contextualized_all
model_names["contextualized_all"]["sft"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/sft/llama-7B_contextualized_all/final_checkpoint"
model_names["contextualized_all"]["dpo"] = "/projects/tir6/general/sachink/personalized-LM/2023/models/0923-newreddit/dpo/llama-7B_contextualized_all/"
model_names["contextualized_all"]["ppo"] = "/projects/tir5/users/sachink/personalized-LM/0923-newreddit/rlhf/llama-se-rl-finetune-128-8-8-1.4e-5_adam_contextualized_allstep_400"

instrtype = script_args.instrtype
subset = script_args.subset
algorithm = script_args.algorithm

model_name = model_names[f'{instrtype}_{subset}'][f"{algorithm}"]

if script_args.single_peft_load:
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
else:
    base_model_name = "/projects/tir6/general/sachink/personalized-LM/2023/llama/hf_model-7B"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_8bit=True)
    model = PeftModel.from_pretrained(base_model, model_name)
# model.eval()
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

generation_kwargs = {"top_k": 0.0, "top_p": 1.0, "do_sample": True, "eos_token_id": -1}
model.eval()
model.to(device)

print(f"device={model.device}")

# %%
# from peft import PeftConfig, PeftModel

# peft_config = PeftConfig.from_pretrained(adapter_model_name)

# model = AutoPeftModelForCausalLM.from_pretrained(
#     base_model_name, return_dict=True, torch_dtype=torch.bfloat16
# )

# # tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# device=0
# if not torch.cuda.is_available():
#     device="cpu"  


# # # Load the Lora model
# model = PeftModel.from_pretrained(model, adapter_model_name).half().to(device)
# model.eval()

# model = model.merge_and_unload()

# model.save_pretrained(f"{script_args.output_name}")
# tokenizer.save_pretrained(f"{script_args.output_name}")
# model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)



# %%
# n = int(input("Output length: " ))
# instrtype = "plain"
# domain = "askphysics"
# domain = "legaladvice"

print("Loading dataset...")
if script_args.subset == "all":
    ds = load_dataset("stanfordnlp/shp", split="test")
else:
    ds = load_dataset("stanfordnlp/shp", data_dir=script_args.subset, split="test")
# if data_dir != "all" and data_dir != "contextual":
# ds = ds.filter(lambda x: x["domain"]==f"{data_dir}_train")
print(f"number of test examples: {len(ds)}")
cols_to_remove = ds.column_names
cols_to_remove.remove("history")
cols_to_remove.remove("post_id")
cols_to_remove.remove("domain")
# print(len(ds), cols_to_remove)
ds = ds.remove_columns(cols_to_remove)
df = ds.to_pandas()

coveredposts = set()

logging.info(f"Done loading data with {ds.shape[0]} entries!")
# data_module = dict(
#     train_dataset=data['train'],
#     eval_dataset=data['validation'],
#     data_collator=DataCollatorForData2TextLanguageModeling(tokenizer=tokenizer),
# )
c = 0

# def clean_text(text: str) -> str:
#     return text.replace("\n", " ")

# print("read the file with", ds.shape[0], "entries")
batchsize = script_args.batch_size
x = 0
st=time.time()
emptylines = []
text_batch = []
posttext_batch = []

os.makedirs(script_args.output_dir, exist_ok=True)
output_file = f"{script_args.output_dir}/{script_args.algorithm}_{script_args.instrtype}_{script_args.subset}.jsonl"
print(output_file)
fout = open(output_file, "w")
# print(fout)
# reward_model_name = script_args.reward_model_name
# reward_model = transformers.T5ForConditionalGeneration.from_pretrained(reward_model_name).to(device)
# reward_model_tokenizer = transformers.T5Tokenizer.from_pretrained(reward_model_name)
y = 0

segmenter = pysbd.Segmenter(language="en", clean=False)
with torch.no_grad():
    for i, row in tqdm(df.iterrows(), total=df.shape[0]): 
        # print(i, batchsize)
    # for i in range(0, len(ds), batchsize):
        # print(i, coveredposts)
        if i == 0 or len(text_batch) < batchsize:
            # print(i, len(text_batch), batchsize)
            if row['post_id'] not in coveredposts:
                # cleanpost = clean_text(row["history
                # "])
                posttext_batch.append(row["history"])
                domain = row['domain'].split("_")[0]
                if instrtype == "subredditname":
                    instruction = f"Write a response to this reddit post in the following subreddit. SUBREDDIT: {domain}. \n\n POST: "
                elif instrtype == "contextualized":
                    instruction = f"Write a response to this reddit post in the subreddit with the following description. SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]}. \n\n POST: "
                else:
                    instruction = f"Write a response to this reddit post. \n\n POST: "

                sentences = []
                slack = 500
                for s in segmenter.segment(row["history"]):
                    l = len(tokenizer(s).input_ids)
                    slack -= l

                    if slack > 0:
                        sentences.append(s)

                text_batch.append(instruction + "".join(sentences) + " \n\n COMMENT: ")
                # if script_args.instrtype == "sft_plain":
                #     text_batch.append(f"POST: {cleanpost}. \n\n What is a good response to this post? RESPONSE: ")
                # elif script_args.instrtype == "sft_contextualized":
                #     domain = row['domain'].split("_")[0]
                #     text_batch.append(f"SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]} \n\n POST: {cleanpost}\n\n What is a good response to this post? RESPONSE: ")
                # elif script_args.instrtype == "sft_subredditname":
                #     domain = row['domain'].split("_")[0]
                #     text_batch.append(f"SUBREDDIT: {domain} \n\n POST: {cleanpost}\n\n What is a good response to this post? RESPONSE: ")
                # elif script_args.instrtype == "plain":
                #     text_batch.append(f"Write a response to this reddit post. \n\n POST: {cleanpost} \n\n RESPONSE: ")
                # elif script_args.instrtype == "contextualized":
                #     domain = row['domain'].split("_")[0]
                #     text_batch.append(f"Write a response to this reddit post in the subreddit with the following description. SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]} \n\n POST: {cleanpost} \n\n RESPONSE: ")
                # elif script_args.instrtype == "subredditname":
                #     domain = row['domain'].split("_")[0]
                #     text_batch.append(f"Write a response to this reddit post in the following subreddit. SUBREDDIT: {domain}. \n\n POST: {cleanpost} \n\n RESPONSE: ")
                # else:
                    # raise ValueError
                    
                coveredposts.add(row['post_id'])
            continue
        
        # print(i, len(text_batch), flush=True)
        # input()
        st = time.time()
        # print(text_batch)
        # tokenizer.padding = "left"
        tokens = tokenizer(text_batch, return_tensors="pt", padding=True).to(device) # check

        # for k in tokens.keys():
        #     tokens[k] = tokens[k].cuda()
        #     torch.cuda.empty_cache()
        # print(time.time() - st)
        st=time.time()
        text_outputs = [{} for _ in range(len(text_batch))]
        # for top_p in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        for top_p in [0.95]:
            generation_kwargs = {"top_p": top_p, "do_sample": True}
            outputs = model.generate(**tokens, max_new_tokens=128, **generation_kwargs) # check
            # print(tokens)
            # print(outputs)
            # input()
            for k, output in enumerate(outputs):
                full_text_output = tokenizer.decode(output, skip_special_tokens=True)
                # print(text_batch[k], "AAAAAAA", full_text_output)
                # input()
                text_outputs[k][top_p] = full_text_output[len(text_batch[k]):]
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

        # input("see")

        x += len(text_outputs)
        y += len(filewrite)
        # print(y)
        text_batch = []
        posttext_batch = []
        # print(posttext_batch)
        # print(text_outputs)

        if i % 50 == 0:
            print(f"done total {y} inputs, took {(time.time()-st)/(x)} seconds per sentence since last log", flush=True)
            x = 0

print(f"wrote {y} lines")

fout.close()



# if instrtype == "subredditname":
#     instruction = f"Write a response to this reddit post in the following subreddit. SUBREDDIT: {domain}. \n\n POST: "
# elif instrtype == "contextualized":
#     instruction = f"Write a response to this reddit post in the subreddit with the following description. SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]}. \n\n POST: "
# else:
#     instruction = f"Write a response to this reddit post. \n\n POST: "

# n = 128

# lines = ""
# no_of_lines = 1
# print(f"Enter new prompt ({no_of_lines} lines): ")
# for i in range(no_of_lines):
#     lines += input()+"\n\n"
    
# # input(batch)
# query_tensors = tokenizer.encode(instruction+lines+" \n\n COMMENT: ", return_tensors="pt").to(device)
# print(query_tensors)

# # query_tensors = tokenizer.encode(lines, return_tensors="pt").to(device)
# print("Output: ", end="")
# # Get response from t5
# response_tensor = model.generate(input_ids=query_tensors, max_new_tokens=n, **generation_kwargs)
# response = tokenizer.decode(response_tensor[0], skip_special_tokens=True)

# print(response)

# # %%



