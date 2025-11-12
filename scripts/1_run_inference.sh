#!/bin/bash

# This script runs the inference script with specified quantization parameters.
# It is intended to be run *inside* the docker container.
#
# Arguments:
# $1: model_name (e.g., "meta-llama/Llama-3.2-3B-Instruct")
# $2: w_bit (e.g., 4)
# $3: a_bit (e.g., 4)
# $4: kv_bit (e.g., 4)
# $5: groupsize (e.g., 128)
#
# Example Usage (from outside container):
# docker compose run --rm spinquant bash scripts/1_run_inference.sh "meta-llama/Llama-3.2-3B-Instruct" 4 4 4 128

MODEL_NAME=$1
W_BITS=$2
A_BITS=$3
KV_BITS=$4
GROUPSIZE=$5

# Construct the dynamic rotation path, mirroring scripts/2_eval_ptq.sh
ROTATION_PATH="/app/models/rotation/${MODEL_NAME}/W${W_BITS}A${A_BITS}KV${KV_BITS}GS${GROUPSIZE}/R.bin"

echo "Using rotation path: ${ROTATION_PATH}"
# Just run the python script directly.
# We are already inside the container.
python run_inference.py \
--input_model ${MODEL_NAME} \
--w_bits ${W_BITS} \
--a_bits ${A_BITS} \
--k_bits ${KV_BITS} \
--v_bits ${KV_BITS} \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize ${GROUPSIZE} \
--v_groupsize ${GROUPSIZE} \
--rotate \
--optimized_rotation_path ${ROTATION_PATH} \
--fp16 False \
--bf16 True \