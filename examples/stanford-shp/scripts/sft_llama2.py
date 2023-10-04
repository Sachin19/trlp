# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

from functools import partial

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
    data_source: Optional[str] = field(default="reddit", metadata={"help": "the model name, write all for all domains combined"})
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    output_dir: Optional[str] = field(default="/projects/tir6/general/sachink/personalized-LM/2023/models/sft", metadata={"help": "the model name"})
    data_dir: Optional[str] = field(default="data", metadata={"help": "the model name, write all for all domains combined"})
    data_prefix: Optional[str] = field(default="sft_", metadata={"help": "the model name, write all for all domains combined"})
    instrtype: Optional[str] = field(default="sft_", metadata={"help": "the model name, write all for all domains combined"})

    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})

    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="all", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    max_steps: Optional[int] = field(default=500, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=True, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=False, metadata={"help": "whether to use packing for SFTTrainer"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    score_ratio_threshold: Optional[float] = field(default=2.0)
    num_examples_per_post: Optional[float] = field(default=5)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir="data/finetune",
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def prepare_sample_text_plain(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    text = example["input_plain"] + example["output"]
    return text

def prepare_sample_text_contextualized(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    text = example["input_contextualized"] + example["output"]
    return text

def prepare_sample_text_subredditname(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    text = example["input_subredditname"] + example["output"]
    return text

prepare_sample_text_fns ={
    "plain": prepare_sample_text_plain,
    "contextualized": prepare_sample_text_contextualized,
    "subredditname": prepare_sample_text_subredditname,
}


def preprocess_function_contextualized(example):
    domain = example['domain'].split("_")[0]
    instruction = f"Write a response to this reddit post in the subreddit with the following description. SUBREDDIT: {SUBREDDIT2DESCRIPTION[domain]}. \n\n POST: "
    
    preferred_output = example['human_ref_A']
    dispreferred_output = example['human_ref_B']
    if example['labels'] == 0:
        preferred_output, dispreferred_output = dispreferred_output, preferred_output

    return instruction + example['history'] + " \n\n COMMENT: " + preferred_output
    # tokenized_question = tokenizer(query, truncation=True)
    # new_examples["query"].append(query)
    # new_examples["input_ids"].append(tokenized_question["input_ids"])

    # return new_examples


def preprocess_function_subredditname(example):
    domain = example['domain'].split("_")[0]
    instruction = f"Write a response to this reddit post in the following subreddit. SUBREDDIT: {domain}. \n\n POST: "
      
    preferred_output = example['human_ref_A']
    dispreferred_output = example['human_ref_B']
    if example['labels'] == 0:
        preferred_output, dispreferred_output = dispreferred_output, preferred_output

    return instruction + example['history'] + " \n\n COMMENT: " + preferred_output
    # tokenized_question = tokenizer(query, truncation=True)
    # new_examples["query"].append(query)
    # new_examples["input_ids"].append(tokenized_question["input_ids"])

    # return new_examples


def preprocess_function_plain(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    # text = example["input_plain"] + example["output"]
    # return text
    # for question, domain in zip(examples["history"], examples['domain']):    
    instruction = f"Write a response to this reddit post. \n\n POST: "

    preferred_output = example['human_ref_A']
    dispreferred_output = example['human_ref_B']
    if example['labels'] == 0:
        preferred_output, dispreferred_output = dispreferred_output, preferred_output

    return instruction + example['history'] + " \n\n COMMENT: " + preferred_output
    # tokenized_question = tokenizer(query, truncation=True)
    # new_examples["query"].append(query)
    # new_examples["input_ids"].append(tokenized_question["input_ids"])

    # return new_examples


preprocess_functions ={
    "plain": preprocess_function_plain,
    "contextualized": preprocess_function_contextualized,
    "subredditname": preprocess_function_subredditname,
}


def chars_token_ratio(dataset, tokenizer, instrtype=None, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        if instrtype is None:
            text = prepare_sample_text(example)
        elif instrtype.startswith("preprocessed"):
            text = prepare_sample_text_fns[instrtype.split("_")[1]](example)
        else:
            text = preprocess_functions[instrtype](example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    print(total_characters, total_tokens)
    return total_characters / total_tokens

def create_datasets_reddit(tokenizer, args):
    if args.data_source == "chp":
        print("loading chp")
        load_dataset = load_from_disk
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_path = args.data_dir
    subset = args.subset
    instrtype = args.instrtype

    def subsample(dataset, ratio_thresh, examples_per_post):
        df = dataset.to_pandas()
        df = df[df["score_ratio"] >=  ratio_thresh]
        df = df.groupby("post_id").apply(
            lambda x: x.sample(n=min(examples_per_post, len(x)))
        )
        df = df.sample(n=len(df))
        return Dataset.from_pandas(df)
    
    if script_args.subset == "all":
        dataset = load_dataset(data_path)
    else:
        dataset = load_dataset(data_path, data_dir=script_args.subset)
    
    print(dataset)
    train_data = dataset["train"]
    valid_data = dataset["validation"]

    print(f"Original training data size: {len(train_data)}")
    train_data = subsample(train_data, script_args.score_ratio_threshold, script_args.num_examples_per_post)
    print(f"Filtered training data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(train_data)}")

    print(f"Validation data size: {len(valid_data)}")

    # original_columns = dataset.column_names
    # num_proc = 24
    
    chars_per_token = chars_token_ratio(train_data, tokenizer, instrtype)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    preprocess_function = preprocess_functions[instrtype]
    
    train_dataset = train_data.map(lambda x: {'text': preprocess_function(x)})
    valid_dataset = valid_data.map(lambda x: {'text': preprocess_function(x)})

    return train_dataset, valid_dataset
    
    # train_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     train_data,
    #     formatting_func=preprocess_function,
    #     infinite=True,
    #     seq_length=args.seq_length,
    #     chars_per_token=chars_per_token,
    # )
    # valid_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     valid_data,
    #     formatting_func=preprocess_function,
    #     infinite=False,
    #     seq_length=args.seq_length,
    #     chars_per_token=chars_per_token,
    # )
    # return train_dataset, valid_dataset

    # ds = train_dataset.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=num_proc,
    #     remove_columns=original_columns,
    # )
    # ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    # ds.set_format(type="torch")
    # return ds

def create_datasets_subreddits_preformatted(tokenizer, args):
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_dir = args.data_dir
    data_prefix = args.data_prefix
    instrtype = args.instrtype
    
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": f"{data_dir}/{data_prefix}train.json", "valid": f"{data_dir}/{data_prefix}validation.json"})
    # dataset.set_format("pandas")

    if args.subset == "all":
        train_data = dataset["train"]
        print(train_data)
        valid_data = dataset["valid"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    else:
        train_data = dataset["train"].filter(lambda example: example['domain'] == args.subset+"_train")
        print(train_data)
        valid_data = dataset["valid"].filter(lambda example: example['domain'] == args.subset+"_validation")
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    # train_data = dataset["train"]
    # print(train_data)
    # valid_data = dataset["valid"]
    # print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    # if args.streaming:
    #     print("Loading the dataset in streaming mode")
    #     valid_data = dataset.take(args.size_valid_set)
    #     train_data = dataset.skip(args.size_valid_set)
    #     train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    # else:
        # dataset = dataset.train_test_split(test_size=0.005, seed=None)
        # train_data = dataset["train"]
        # valid_data = dataset["test"]
        # print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, instrtype)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text_fns[instrtype],
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text_fns[instrtype],
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    bf16=True,
    remove_unused_columns=False,
    run_name="sft_llama2",
    ddp_find_unused_parameters=False,
)

if script_args.data_source == "SO":
    train_dataset, eval_dataset = create_datasets(tokenizer, script_args)
else:
    train_dataset, eval_dataset = create_datasets_reddit(tokenizer, script_args)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    packing=script_args.packing,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(script_args.output_dir)

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
