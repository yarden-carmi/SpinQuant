# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py \
--input_model $1  \
--output_rotation_path "/app/models/rotation/$1/ETW${2}A${3}KV${4}GS${5}/" \
--output_dir "/app/output/$1/ETW${2}A${3}KV${4}GS${5}/" \
--logging_dir "/app/logs/$1/ETW${2}A${3}KV${4}GS${5}/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
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
--v_groupsize $5
