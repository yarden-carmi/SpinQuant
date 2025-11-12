import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.process_args import process_args_ptq
from eval_utils.rotation_utils import rotate_model
from train_utils.main import prepare_model

# ---
# This script mimics the model loading logic from ptq.py
# ---

# !!!
# Using a model-specific path structure as you suggested.
# !!!
MODEL_NAME = "Llama-3.2-3B-Instruct"
MODEL_ID = f"meta-llama/{MODEL_NAME}"
# Path is now dynamically set based on the MODEL_NAME
OPTIMIZED_ROTATION_PATH = "/app/models/rotation/meta-llama/Llama-3.2-3B-Instruct/R.bin"

print("--- Starting Inference Script ---")

# 1. Manually set ALL arguments
# We have to spoof sys.argv for the process_args_ptq function
sys.argv = [
    'run_inference.py',
    '--input_model', MODEL_ID, # Use variable
    '--w_bits', '4',
    '--a_bits', '4',
    '--k_bits', '4',
    '--v_bits', '4',
    '--w_clip',
    '--a_asym',
    '--k_asym',
    '--v_asym',
    '--k_groupsize', '128',
    '--v_groupsize', '128',
    '--rotate',
    '--optimized_rotation_path', OPTIMIZED_ROTATION_PATH, # Use the corrected path
    '--bf16',
]

# 2. Parse all arguments using the correct function
# This will create all three objects: model_args, training_args, and ptq_args
model_args, training_args, ptq_args = process_args_ptq()

print(f"Loading base model: {model_args.input_model}")
# 3. Load base tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_args.input_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_args.input_model,
    device_map='cuda:0',  # Use all available GPUs
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("Applying rotation and quantization...")
# 4. Apply the SpinQuant rotation (R.bin)
#model = rotate_model(model, ptq_args) # ptq_args has the rotation path

# 5. Apply the W4A4KV4 quantization wrappers
model = prepare_model(ptq_args, model)
model.to('cuda:0')
model.eval()

print("--- Model is quantized and ready ---")

# 6. Run inference
prompt = "The capital of Israel is"
messages = [
    {"role": "user", "content": prompt},
]

# Apply the Llama 3.2 instruct chat template
prompt_formatted = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)

# Generate output
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