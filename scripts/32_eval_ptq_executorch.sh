# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
OUTPUT_DIR="/home/jetson/Desktop/Models/models/output/$1/ETW${2}A${3}KV${4}GS${5}/"
mkdir -p $OUTPUT_DIR

torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--w_groupsize $5 \
--k_groupsize $5 \
--v_groupsize $5 \
--rotate \
--optimized_rotation_path "/home/jetson/Desktop/Models/models/rotation/$1/ETW${2}A${3}KV${4}GS${5}/R.bin" \
--save_qmodel_path "${OUTPUT_DIR}consolidated.00.pth" \
--export_to_et
