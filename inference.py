# Description: Load a base model and fine-tuned LoRA adapters for inference using Hugging Face Transformers and PEFT.
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# ---------- 1. Define model identifiers and paths ----------
# Base pretrained model identifier from Hugging Face Hub.
# Change this if you used a different base model for QLoRA/LoRA training.
base_model_name = "mistralai/Mistral-7B-v0.1"

# Path where the fine-tuned LoRA adapter files were saved (from training step).
lora_model_path = "./mistral-qlora-final"

# ---------- 2. Load tokenizer ----------
# Tokenizer converts raw text to token IDs and back. We load the base model tokenizer
# because the LoRA adapters were trained with the same tokenizer.
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# ---------- 3. Load base model (quantized for memory efficiency) ----------
# We load the base causal LM in 4-bit mode (QLoRA) to reduce memory use.
# - load_in_4bit=True: use 4-bit quantization (requires bitsandbytes + compatible transformers)
# - device_map="auto": automatically place model layers on available devices (GPUs/CPU)
# - torch_dtype=torch.float16: use FP16 for compute where supported (mixed precision)
# Note: Ensure your environment supports 4-bit loading and the required libraries are installed.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,         # QLoRA: 4-bit quantized weights to save RAM
    device_map="auto",         # place layers automatically across devices
    torch_dtype=torch.float16, # use FP16 for faster compute when supported
)

# ---------- 4. Load LoRA adapters and combine for inference ----------
# PeftModel.from_pretrained wraps the base_model and loads LoRA adapter weights from lora_model_path.
# The resulting 'model' still uses adapter logic (PEFT) during inference; it does not permanently
# modify the base model weights unless you explicitly merge and save them.
model = PeftModel.from_pretrained(base_model, lora_model_path)

# ---------- 5. Create a text-generation pipeline for inference ----------
# The transformers.pipeline handles tokenization, model execution and decoding.
# Passing the PEFT-wrapped model returns generations that include the adapter effects.
# device_map="auto" inside pipeline isn't a standard pipeline arg — the pipeline will respect
# model.device placement done above. If you need to force device, move model.to(device) first.
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",  # keep; model layers were already placed by device_map when loading
)

# ---------- 6. Run a short generation (example prompt) ----------
# - max_new_tokens: maximum number of tokens to generate
# - temperature: sampling temperature (higher = more random)
# - top_p: nucleus sampling parameter (probability mass to consider)
# - do_sample: sample from distribution instead of greedy decoding
prompt = "The food at the restaurant was"
outputs = pipe(
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# The pipeline returns a list of generation results; print the generated text of the first result.
print(outputs[0]["generated_text"])
