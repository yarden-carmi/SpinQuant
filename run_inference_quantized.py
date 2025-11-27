import torch
import sys
from transformers import AutoTokenizer, LlamaTokenizerFast
import transformers
from eval_utils.modeling_llama import LlamaForCausalLM
from utils.process_args import process_args_ptq
from eval_utils.main import ptq_model
from utils import utils

# ---
# This script loads a quantized model and runs a sample inference.
# ---

print("--- Starting Inference Script (Quantized) ---")

# 1. Parse arguments from command line
try:
    model_args, training_args, ptq_args = process_args_ptq()
except Exception as e:
    print(f"Error parsing arguments: {e}")
    sys.exit(1)

print(f"Loading base model: {model_args.input_model}")

# 2. Load base tokenizer and model
config = transformers.AutoConfig.from_pretrained(
    model_args.input_model, token=model_args.access_token
)
# Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
process_word_embeddings = False
if config.tie_word_embeddings:
    config.tie_word_embeddings = False
    process_word_embeddings = True
dtype = torch.bfloat16 if training_args.bf16 else torch.float16
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_args.input_model,
    config=config,
    torch_dtype=dtype,
    token=model_args.access_token,
)
if process_word_embeddings:
    model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
model.cuda()

print("Applying rotation and quantization wrappers and loading quantized weights...")
# 3. Apply rotation and quantization wrappers and load weights
# Ensure load_qmodel_path is set in arguments
if not ptq_args.load_qmodel_path:
    print("Warning: --load_qmodel_path not provided. Model will be quantized on the fly (fake quantization) if w_bits < 16.")

model = ptq_model(ptq_args, model, model_args)
model.seqlen = training_args.model_max_length
model.eval()
print("--- Model is ready ---")

tokenizer = LlamaTokenizerFast.from_pretrained(
    pretrained_model_name_or_path=model_args.input_model,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=True,
    add_eos_token=False,
    add_bos_token=False,
    token=model_args.access_token,
)

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
