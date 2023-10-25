# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from functools import partial

import torch
from datasets import Dataset, load_dataset, load_from_disk
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer

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

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    data_source: Optional[str] = field(default="SO", metadata={"help": "the model name, write all for all domains combined"})
    instrtype: Optional[str] = field(default=None, metadata={"help": "plain, subredditname, or contextualized"})
    subset: Optional[str] = field(default="all", metadata={"help": "all or subredditname"})

    score_ratio_threshold: Optional[float] = field(default=2.0)
    num_examples_per_post: Optional[float] = field(default=5)

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    data_dir: Optional[str] = field(default=None, metadata={"help": "the output directory"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

def subsample(dataset, ratio_thresh, examples_per_post):
    df = dataset.to_pandas()
    df = df[df["score_ratio"] >=  ratio_thresh]
    df = df.groupby("post_id").apply(
        lambda x: x.sample(n=min(examples_per_post, len(x)))
    )
    df = df.sample(n=len(df))
    return Dataset.from_pandas(df)


def get_chp_paired(
    data_dir: str = "chp",
    split: str = "train",
    subset: str = "all",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    dataset = load_from_disk(data_dir)[split]
    if not subset == "all":
        dataset = dataset.filter(lambda x: x['domain'] == subset)
    
    cols_to_remove = dataset.column_names
    cols_to_remove.remove("history")
    cols_to_remove.remove("human_ref_A")
    cols_to_remove.remove("human_ref_B")
    cols_to_remove.remove("labels")
    cols_to_remove.remove("score_ratio")

    dataset.remove_columns(cols_to_remove)

    print(f"Original training data size: {len(dataset)}")
    dataset = subsample(dataset, script_args.score_ratio_threshold, script_args.num_examples_per_post)
    print(f"Filtered training data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(dataset)}")

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return_object = {"prompt": [], "chosen": [], "rejected": []}
        
        for domain, question, response_j, response_k, label in zip(samples['domain'], samples["history"], samples["human_ref_A"], samples["human_ref_B"], samples['labels']):
            domain = domain.split("_")[0]
            if script_args.instrtype == "subredditname":
                instruction = f"Write a response to this reddit post in the following subreddit. SUBREDDIT: {domain}. \n\n POST: "
            elif script_args.instrtype == "contextualized":
                instruction = f"Write a response to this reddit post in the subreddit with the following description. SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]}. \n\n POST: "
            else:
                instruction = f"Write a response to this reddit post. \n\n POST: "

            if label == 0:
                response_j, response_k = response_k, response_j
            
            prompt = instruction + question + " \n\n COMMENT: "
            return_object['prompt'].append(prompt)
            return_object['chosen'].append(response_j)
            return_object['rejected'].append(response_k)

        return return_object
    
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


def get_reddit_paired(
    data_dir: str = "stanfordnlp/shp",
    split: str = "train",
    subset: str = "all",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    if subset == "all":
        dataset = load_dataset(data_dir, split=split, cache_dir=cache_dir)
    else:
        dataset = load_dataset(data_dir, split=split, data_dir=subset, cache_dir=cache_dir)
    
    cols_to_remove = dataset.column_names
    cols_to_remove.remove("history")
    cols_to_remove.remove("human_ref_A")
    cols_to_remove.remove("human_ref_B")
    cols_to_remove.remove("labels")
    cols_to_remove.remove("score_ratio")

    dataset.remove_columns(cols_to_remove)

    print(f"Original training data size: {len(dataset)}")
    dataset = subsample(dataset, script_args.score_ratio_threshold, script_args.num_examples_per_post)
    print(f"Filtered training data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(dataset)}")

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return_object = {"prompt": [], "chosen": [], "rejected": []}
        
        for domain, question, response_j, response_k, label in zip(samples['domain'], samples["history"], samples["human_ref_A"], samples["human_ref_B"], samples['labels']):
            domain = domain.split("_")[0]
            if script_args.instrtype == "subredditname":
                instruction = f"Write a response to this reddit post in the following subreddit. SUBREDDIT: {domain}. \n\n POST: "
            elif script_args.instrtype == "contextualized":
                instruction = f"Write a response to this reddit post in the subreddit with the following description. SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]}. \n\n POST: "
            else:
                instruction = f"Write a response to this reddit post. \n\n POST: "

            if label == 0:
                response_j, response_k = response_k, response_j
            
            prompt = instruction + question + " \n\n COMMENT: "
            return_object['prompt'].append(prompt)
            return_object['chosen'].append(response_j)
            return_object['rejected'].append(response_k)

        return return_object
    
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    # print(script_args.model_name_or_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.config.use_cache = False
    
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoPeftModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    if script_args.data_source == "SO":
        # 2. Load the Stack-exchange paired dataset
        train_dataset = get_stack_exchange_paired(data_dir="data/rl", sanity_check=script_args.sanity_check)
        train_dataset = train_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
            and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
        )

        # 3. Load evaluation dataset
        eval_dataset = get_stack_exchange_paired(data_dir="data/evaluation", sanity_check=True)
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
            and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
        )
    elif script_args.data_source == "chp":
        # 2. Load the Stack-exchange paired dataset
        train_dataset = get_chp_paired(data_dir=script_args.data_dir, split="train", subset=script_args.subset, sanity_check=script_args.sanity_check)
        train_dataset = train_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
            and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
        )

        # 3. Load evaluation dataset
        eval_dataset = get_chp_paired(data_dir=script_args.data_dir, split="validation", subset=script_args.subset, sanity_check=True)
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
            and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
        )

    else:
        # 2. Load the Stack-exchange paired dataset
        train_dataset = get_reddit_paired(data_dir="stanfordnlp/shp", split="train", subset=script_args.subset, sanity_check=script_args.sanity_check)
        train_dataset = train_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
            and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
        )

        # 3. Load evaluation dataset
        eval_dataset = get_reddit_paired(data_dir="stanfordnlp/shp", split="validation", subset=script_args.subset, sanity_check=True)
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
            and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
        )


    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_llama2",
        ddp_find_unused_parameters=False,
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
