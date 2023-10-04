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

import time
import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForSequenceClassification, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, Pipeline, pipeline, BitsAndBytesConfig
from transformers.pipelines import PIPELINE_REGISTRY

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

tqdm.pandas()

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




class RewardPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        return self.tokenizer(inputs, return_tensors=self.framework)
        # model_input = torch.Tensor(inputs["input_ids"])
        # return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        
        return {"score": logits}


PIPELINE_REGISTRY.register_pipeline(
    "reward",
    pipeline_class=RewardPipeline,
    pt_model=AutoPeftModelForSequenceClassification,
)

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    data_source: Optional[str] = field(default="SO", metadata={"help": "the model name"})
    subset: Optional[str] = field(default="all", metadata={"help": "the model name"})
    instrtype: Optional[str] = field(default="plain", metadata={"help": "the model name"})

    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    eval_freq: Optional[int] = field(default=None, metadata={"help": "n steps to eval the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})

    score_ratio_threshold: Optional[float] = field(default=2.0)
    num_examples_per_post: Optional[float] = field(default=5)

def subsample(dataset, ratio_thresh, examples_per_post):
    df = dataset.to_pandas()
    df = df[df["score_ratio"] >=  ratio_thresh]
    df = df.groupby("post_id").apply(
        lambda x: x.sample(n=min(examples_per_post, len(x)))
    )
    df = df.sample(n=len(df))
    return Dataset.from_pandas(df)

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
dataset_name = "lvwerra/stack-exchange-paired"
config = PPOConfig(
    tracker_project_name=f"ppo/{script_args.model_name.replace('/', '-')}_{script_args.instrtype}_{script_args.subset}"[-128:],
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset_reddit(
    tokenizer, 
    dataset,
    instrtype):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # dataset = subsample(dataset, script_args.score_ratio_threshold, script_args.num_examples_per_post)
    # load imdb with datasets
    #ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    original_columns = dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question, domain in zip(examples["history"], examples['domain']):
            domain = domain.split("_")[0]
            if instrtype == "subredditname":
                instruction = f"Write a response to this reddit post in the following subreddit. SUBREDDIT: {domain}. \n\n POST: "
            elif instrtype == "contextualized":
                instruction = f"Write a response to this reddit post in the subreddit with the following description. SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]}. \n\n POST: "
            else:
                instruction = f"Write a response to this reddit post. \n\n POST: "

            query = instruction + question + " \n\n COMMENT: "
            tokenized_question = tokenizer(query)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds

def build_dataset_SO(
    tokenizer,
    ds
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load imdb with datasets
    #ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if script_args.data_source == "SO":
    train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
    train_dataset = train_dataset.select(range(100000))

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset_SO(tokenizer, train_dataset)
else:
    if script_args.subset == "all":
        train_dataset = load_dataset("stanfordnlp/shp", split="train")
        eval_dataset = load_dataset("stanfordnlp/shp", split="validation")
    else:
        train_dataset = load_dataset("stanfordnlp/shp", split="train", data_dir=script_args.subset)
        eval_dataset = load_dataset("stanfordnlp/shp", split="validation", data_dir=script_args.subset)
    
    print(f"Original training data size: {len(train_dataset)}")
    print(f"Original eval data size: {len(eval_dataset)}")
    train_dataset = subsample(train_dataset, script_args.score_ratio_threshold, script_args.num_examples_per_post)
    print(f"Filtered training data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(train_dataset)}")

    dataset = build_dataset_reddit(tokenizer, train_dataset, script_args.instrtype)
    eval_dataset = build_dataset_reddit(tokenizer, eval_dataset, script_args.instrtype)
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}


if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_4bit_use_double_quant=False,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    # quantization_config=bnb_config,
    device_map={"": current_device},
 #   peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    optimizer=optimizer,
    # run_name="ppo_llama2"
)

# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
print(device)
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

print(device)
print(current_device)
reward_model = AutoPeftModelForSequenceClassification.from_pretrained(
                                                            reward_model_name, 
                                                            num_labels=1, 
                                                            # quantization_config=bnb_config)
                                                            load_in_8bit=True)#torch_dtype=torch.bfloat16)
reward_model.eval()
# sentiment_pipe = pipeline(
#     "reward",
#     model=reward_model,
#     device_map={"": current_device},
#     model_kwargs={"load_in_8bit": True},
#     tokenizer=tokenizer,
#     return_token_type_ids=False,
# )

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

print(config.total_ppo_epochs)
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]
    # print(len(question_tensors))
    s = time.time()
    with torch.no_grad():

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        s2 = time.time()
        print(epoch, (s2 - s))
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        s3 = time.time()
        print(s3-s2)
        # Compute reward score (using the sentiment analysis pipeline)
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        print(time.time() - s3)
        reward_model_input = tokenizer(texts, padding=True, return_tensors="pt")
        # print(reward_model_input)
        # print(response_tensors)
        # print(question_tensors)
        #input()
        
        pipe_outputs = reward_model(**reward_model_input).logits
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        # print(pipe_outputs, 
        print(time.time()-s3)
        rewards = list(pipe_outputs)
    # rewards = [torch.tensor(output["score"] - script_args.reward_baseline) for output in pipe_outputs]
    
        # print(rewards)
        # input("look")
    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # print(stats)
    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")

    print(script_args.eval_freq, epoch)
    if script_args.eval_freq and epoch and epoch % script_args.eval_freq == 0:
        print(f"evaluation {len(ppo_trainer.eval_dataloader)}")
        all_rewards = []
        for eval_batch in tqdm(ppo_trainer.eval_dataloader):
            question_tensors = eval_batch["input_ids"]
            # print(len(question_tensors))
            s = time.time()
            with torch.no_grad():

                response_tensors = ppo_trainer.generate(
                    question_tensors,
                    return_prompt=False,
                    length_sampler=output_length_sampler,
                    **generation_kwargs,
                )
                s2 = time.time()
                # print(epoch, (s2 - s))
                eval_batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                s3 = time.time()
                # print(s3-s2)
                # Compute reward score (using the sentiment analysis pipeline)
                texts = [q + r for q, r in zip(eval_batch["query"], eval_batch["response"])]
                # print(time.time() - s3)
                reward_model_input = tokenizer(texts, padding=True, return_tensors="pt")
                # print(reward_model_input)
                # print(response_tensors)
                # print(question_tensors)
                #input()
                
                pipe_outputs = reward_model(**reward_model_input).logits
            # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
                # print(pipe_outputs, 
                # print(time.time()-s3)
                rewards = pipe_outputs
                all_rewards.append(rewards)
            # rewards = [torch.tensor(output["score"] - script_args.reward_baseline) for output in pipe_outputs]
            
                # print(rewards)
                # input("look")
            # Run PPO step
            # stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_eval_stats(stats, torch.cat(all_rewards, dim=0)) # check

            
