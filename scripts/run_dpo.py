import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import wandb
import logging
logging.getLogger("wandb").setLevel(logging.ERROR)


# Model and run setup
MODEL_NAME = "sshleifer/tiny-gpt2"  # swap with "mistralai/Mistral-7B-v0.1" for real runs
PROJECT_NAME = "dpo-mistral-test"
RUN_NAME = "mistral-dpo-run"

# Init wandb
wandb.init(project=PROJECT_NAME, name=RUN_NAME)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Avoid padding errors
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

# Load and preprocess dataset
dataset = load_dataset("Anthropic/hh-rlhf")["train"].select(range(1000))

def flatten(example):
    return {
        # "prompt": example["prompt"],  # or "" if you don't have it
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

def tokenize_fn(example):
    chosen = tokenizer(
        example["chosen"],
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    rejected = tokenizer(
        example["rejected"],
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    return {
        "chosen_input_ids": chosen["input_ids"],
        "rejected_input_ids": rejected["input_ids"],
    }

dataset = dataset.map(flatten)
dataset = dataset.map(tokenize_fn)

   

# DPO Config
config = DPOConfig(
    beta=0.1,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=100,
    output_dir="./dpo_output",
    num_train_epochs=1,
    report_to="wandb",
    remove_unused_columns=False,
)

# Initialize trainer (note: no tokenizer here!)
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=config,
    train_dataset=dataset,
)

# Train
trainer.train()
