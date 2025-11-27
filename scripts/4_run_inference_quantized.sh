export CUDA_VISIBLE_DEVICES=0

# Absolute paths
MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
ROTATION_PATH="/home/jetson/Desktop/Models/models/rotation/meta-llama/Llama-3.2-1B-Instruct/ETW4A4KV4GS64/R.bin"
QMODEL_PATH="/home/jetson/Desktop/Models/models/output/meta-llama/Llama-3.2-1B-Instruct/ETW4A4KV4GS64/consolidated.00.pth"

echo "Running inference with quantized model..."
echo "Model: $MODEL_PATH"
echo "Rotation: $ROTATION_PATH"
echo "Quantized Checkpoint: $QMODEL_PATH"

python run_inference_quantized.py \
    --input_model $MODEL_PATH \
    --optimized_rotation_path $ROTATION_PATH \
    --rotate \
    --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 \
    --w_groupsize 64 --a_groupsize 64 \
    --w_clip \
    --model_max_length 2048
