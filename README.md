# QLoRA - Mistral 7B Fine-tuning + Merge + Inference

Short guide for this repository which demonstrates QLoRA + LoRA fine-tuning on a Mistral-7B base model, merging LoRA adapters into the base model, and running inference.

## Repository layout
- `main.py` — training script: load base model in 4-bit (QLoRA), attach LoRA adapters, fine-tune, and save adapters to `./mistral-qlora-final`.
- `merge-LoRA.py` — loads base model + saved LoRA adapters, merges adapters into base weights and saves merged model to `./mistral-qlora-merged`.
- `inference.py` — loads base model + LoRA adapters for runtime inference (does not permanently merge), runs a short generation example.
- `README.md` — this file.

All inline comments in the code are in English for clarity.

## Requirements / prerequisites
- Linux with a CUDA-enabled GPU (recommended). 4-bit loading and bitsandbytes require a compatible CUDA/driver setup.
- Python 3.9+ (adjust if necessary).
- Hugging Face account and access permission for the chosen base model if required.

Suggested Python packages:
- transformers
- peft
- bitsandbytes
- accelerate
- datasets
- torch
- safetensors
- huggingface_hub

