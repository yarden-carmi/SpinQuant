import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.process_args import process_args_ptq
# from eval_utils.rotation_utils import rotate_model  <--- REMOVED THIS IMPORT
from train_utils.main import prepare_model

# ---
# This script loads a model with optimized rotation and quantization
# and runs a sample inference.
# It is intended to be called with arguments, similar to ptq.py
# ---

print("--- Starting Inference Script ---")

# 1. Parse arguments from command line
# We no longer spoof sys.argv. We let process_args_ptq parse them.
try:
    model_args, training_args, ptq_args = process_args_ptq()
except Exception as e:
    print(f"Error parsing arguments: {e}")
    print("Please ensure all required arguments are provided (e.g., --input_model, --w_bits, etc.)")
    sys.exit(1)

print(f"Loading base model: {model_args.input_model}")

# 2. Load base tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_args.input_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_args.input_model,
    device_map='cuda:0',  # Use all available GPUs
    torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16, # Use parsed arg
    trust_remote_code=True,
)

# --- ENTIRE ROTATION BLOCK REMOVED ---
# The prepare_model function handles both rotation and quantization.

print("Applying rotation and quantization wrappers...")
# 3. Apply rotation and quantization wrappers
model = prepare_model(ptq_args, model)
model.to('cuda:0')
model.eval()

print("--- Model is quantized and ready ---")

# 4. Run inference
prompt = "what is the capital of Israel ?"

# Conditionally apply chat template only if it exists
if tokenizer.chat_template:
    print("Applying chat template (Instruct/Chat model detected).")
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Apply the chat template
    prompt_formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
else:
    print("No chat template found (Base model detected). Using raw prompt.")
    prompt_formatted = prompt

inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)

# 5. Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
response = outputs[0][inputs["input_ids"].shape[-1]:]
print(f"\nPrompt: {prompt}")
print(f"Response: {tokenizer.decode(response, skip_special_tokens=True)}")