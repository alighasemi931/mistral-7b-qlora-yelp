from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ---------- 1. Load base model ----------
# Specify the base pretrained model and the path where the trained LoRA weights were saved.
base_model_name = "mistralai/Mistral-7B-v0.1"
lora_model_path = "./mistral-qlora-final"

# Load tokenizer for the base model. We will reuse this tokenizer for the merged model.
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the base model in 4-bit mode (QLoRA) to keep memory usage low.
# device_map="auto" will place layers on available devices automatically.
# torch_dtype=float16 is used for compute efficiency where supported.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,           # load with QLoRA (4-bit quantization)
    device_map="auto",
    torch_dtype=torch.float16,
)

# ---------- 2. Load LoRA ----------
# Wrap the base model with the LoRA configuration saved at lora_model_path.
# PeftModel.from_pretrained will load the LoRA adapters and keep them separate from base weights.
model = PeftModel.from_pretrained(base_model, lora_model_path)

# ---------- 3. Merge LoRA weights into base model ----------
# merge_and_unload() applies the LoRA adapter weights into the base model weights
# and unloads the PEFT wrappers, producing a single merged model that no longer requires PEFT at inference.
merged_model = model.merge_and_unload()

# ---------- 4. Save merged model ----------
# Save the merged model and tokenizer to a new directory so it can be loaded as a normal pretrained model.
save_path = "./mistral-qlora-merged"
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Merged model saved at {save_path}")

# ---------- 5. Quick local test with Transformers pipeline ----------
# Reload the merged model from the saved path and run a short generation to sanity-check the merge.
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_path = "./mistral-qlora-merged"

# Load tokenizer and merged model (now a regular pretrained model).
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Create a text-generation pipeline and generate a short completion from a prompt.
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "The food at the restaurant was"
outputs = pipe(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True)

print(outputs[0]["generated_text"])
