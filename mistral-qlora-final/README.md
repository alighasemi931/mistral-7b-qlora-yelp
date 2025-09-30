---
base_model: mistralai/Mistral-7B-v0.1
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:mistralai/Mistral-7B-v0.1
- lora
- transformers
---

# Mistral-7B + LoRA adapters (QLoRA workflow)

This folder contains metadata and a model card for the LoRA adapters produced by fine-tuning a Mistral-7B base model with QLoRA + PEFT. It documents intended use, quick start commands, requirements, and known limitations.

## Summary
- Type: LoRA adapter weights for a Mistral-7B causal language model.
- Produced by: fine-tuning with PEFT (LoRA) on a base Mistral-7B model using QLoRA quantized base weights.
- Artifacts in this repo:
  - LoRA adapter files and config (saved by model.save_pretrained(...))
  - Tokenizer files (shared with base model)

## Model details
- Base model: mistralai/Mistral-7B-v0.1
- Adapter type: LoRA (PEFT)
- Task: causal text generation (text-generation pipeline)
- Language: English (adapter training data dependent)

## Intended use
- Use to apply the learned adapter weights on top of the base Mistral-7B model for inference or further fine-tuning.
- Two common workflows:
  1. Runtime PEFT (no merge): load the base model (optionally quantized) and wrap it with adapters via PeftModel.from_pretrained(...) — small adapter files are applied at runtime.
  2. Merge adapters: permanently merge LoRA weights into the base weights (merge_and_unload()) and save a single pretrained model for standard inference (no PEFT wrapper required).

Do not use this adapter for tasks it was not trained for without evaluation.

