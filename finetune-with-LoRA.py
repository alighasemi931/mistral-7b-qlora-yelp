# Description: Fine-tune a base causal language model using QLoRA (4-bit quantization + LoRA adapters)
# with Hugging Face Transformers and PEFT. This script demonstrates loading a base model,
# configuring LoRA, preparing a dataset, and running training with the Trainer API.
# After training, only the LoRA adapter weights are saved for efficient storage and later use.
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# ---------- 1. Load model ----------
# Choose the base pretrained model identifier from Hugging Face Hub.
# This example uses Mistral 7B; change to your preferred base model if needed.
model_name = "mistralai/Mistral-7B-v0.1"

# Load the tokenizer for the base model.
# Set pad_token to eos_token because some models do not define a pad token and
# training/evaluation code expects one to exist.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # avoid padding-related errors

# Load the base causal LM. Using 4-bit quantization (QLoRA) reduces memory usage.
# - load_in_4bit=True: enable 4-bit weights (requires bitsandbytes and compatible setup).
# - device_map="auto": automatically place model layers on available devices (GPU/CPU).
# - torch_dtype=torch.float16: use FP16 for compute where supported (mixed precision).
# Note: Ensure your environment supports 4-bit loading (correct bitsandbytes, transformers, accelerate versions).
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,          # load with QLoRA (4-bit quantization)
    device_map="auto",
    torch_dtype=torch.float16,  # use FP16 for compute efficiency
)

# ---------- 2. LoRA configuration ----------
# Configure LoRA (Low-Rank Adaptation) parameters.
# LoRA injects small trainable adapters into the base model so only adapter weights
# are updated during fine-tuning (much cheaper than full fine-tuning).
peft_config = LoraConfig(
    r=8,                                # low-rank dimension
    lora_alpha=16,                      # scaling factor for LoRA updates
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # which modules to adapt (attention projections)
    lora_dropout=0.05,                  # dropout applied to LoRA layers
    bias="none",                        # how to handle bias terms ("none" keeps biases in base model)
    task_type="CAUSAL_LM",              # task type for PEFT/LoRA
)

# Wrap the base model with PEFT so training only updates LoRA adapter weights.
model = get_peft_model(model, peft_config)

# ---------- 3. Prepare dataset ----------
# Load a dataset. Here we use a Hugging Face dataset as example (Yelp reviews).
# Replace this with your own dataset if desired.
dataset = load_dataset("yelp_review_full")  # replace with your own dataset if needed

# Tokenization function used to convert raw text into model inputs.
# - truncation=True: truncate sequences longer than max_length
# - padding="max_length": pad to max_length (useful for batching, but increases memory)
# - max_length=512: maximum sequence length (adjust according to model and GPU memory)
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # For causal language modeling, labels are typically the same as input_ids.
    # Copy input_ids to labels so Trainer computes loss against the input tokens.
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply tokenization to the dataset in batched mode for speed.
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Select small subsets for quick testing/training. Adjust or remove .select() in real runs.
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
eval_dataset = tokenized_datasets["test"].select(range(500))

# ---------- 4. Training settings ----------
# Configure Hugging Face Trainer training arguments.
# Key settings:
# - per_device_train_batch_size: batch size per GPU (set small for limited memory GPUs)
# - gradient_accumulation_steps: simulate larger batch by accumulating gradients
# - fp16: enable mixed precision training (saves memory / speeds up) when supported
training_args = TrainingArguments(
    output_dir="./mistral-qlora-out",
    per_device_train_batch_size=1,   # small per-GPU batch for common GPUs (e.g., 3090)
    gradient_accumulation_steps=8,   # accumulate to reach an effective batch size of 8
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    fp16=True,   # enable FP16 if GPU supports it
    bf16=False,  # enable bf16 only on compatible hardware (like Ampere with proper drivers)
    report_to="none",  # disable external experiment loggers by default
)

# ---------- 5. Run training ----------
# Create a Trainer instance and launch training.
# Trainer handles data loading, optimization loop, evaluation, and checkpointing.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start the training loop. This will only update LoRA adapter weights (not full model).
trainer.train()

# ---------- 6. Save LoRA ----------
# Save only the PEFT/LoRA adapters and tokenizer to a directory so they can be reloaded later.
# The saved folder './mistral-qlora-final' will contain adapter weights and tokenizer files.
model.save_pretrained("./mistral-qlora-final")
tokenizer.save_pretrained("./mistral-qlora-final")
print("LoRA model and tokenizer saved to ./mistral-qlora-final")
# Note: The base model weights are not saved here to keep storage low.