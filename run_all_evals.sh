#!/bin/bash

# This script runs the full 3-step pipeline (optimize, eval, inference)
# for a list of models and quantization parameters.
# All output is logged to evaluation_results.log

# Define the log file
LOG_FILE="evaluation_results.log"

# Clear the log file for a fresh run
> $LOG_FILE

echo "--- Starting Batch Evaluation ---" | tee -a $LOG_FILE
echo "Results will be saved to $LOG_FILE"

# Define models and parameters
# Format: MODEL_NAME W_BIT A_BIT KV_BIT GROUPSIZE

read -r -d '' MODELS_TO_RUN << EOM
meta-llama/Llama-3.2-1B-Instruct 4 4 4 64
EOM

# Loop through each line of the heredoc
while IFS= read -r line; do
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi

    # Read parameters into variables
    read -r MODEL_NAME W_BIT A_BIT KV_BIT GROUPSIZE <<< "$line"

    echo "==========================================================================" | tee -a $LOG_FILE
    echo "STARTING: $MODEL_NAME (W${W_BIT}A${A_BIT}KV${KV_BIT}GS${GROUPSIZE})" | tee -a $LOG_FILE
    echo "==========================================================================" | tee -a $LOG_FILE

    # --- Step 1: Optimize Rotation ---
    echo "\n--- Running 10_optimize_rotation.sh ---" | tee -a $LOG_FILE
    # Append all output (stdout & stderr) to the log file
    docker compose run --rm -T spinquant bash -c "bash scripts/10_optimize_rotation.sh $MODEL_NAME $W_BIT $A_BIT $KV_BIT $GROUPSIZE" >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR during 10_optimize_rotation.sh for $MODEL_NAME. Check $LOG_FILE. Skipping rest for this model." | tee -a $LOG_FILE
        continue
    fi

    # --- Step 2: Evaluate PTQ (Perplexity) ---
    echo "\n--- Running 2_eval_ptq.sh ---" | tee -a $LOG_FILE
    docker compose run --rm -T spinquant bash -c "bash scripts/2_eval_ptq.sh $MODEL_NAME $W_BIT $A_BIT $KV_BIT $GROUPSIZE" >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR during 2_eval_ptq.sh for $MODEL_NAME. Check $LOG_FILE. Skipping inference." | tee -a $LOG_FILE
        continue
    fi

    # --- Step 3: Run Inference (Prompt Response) ---
    echo "\n--- Running 1_run_inference.sh ---" | tee -a $LOG_FILE
    docker compose run --rm -T spinquant bash -c "bash scripts/1_run_inference.sh $MODEL_NAME $W_BIT $A_BIT $KV_BIT $GROUPSIZE" >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR during 1_run_inference.sh for $MODEL_NAME. Check $LOG_FILE." | tee -a $LOG_FILE
    fi

    # --- Step 4: Optimize Rotation (Executorch) ---
    echo "\n--- Running 31_optimize_rotation_executorch.sh ---" | tee -a $LOG_FILE
    docker compose run --rm -T spinquant bash -c "bash scripts/31_optimize_rotation_executorch.sh $MODEL_NAME $W_BIT $A_BIT $KV_BIT $GROUPSIZE" >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR during 31_optimize_rotation_executorch.sh for $MODEL_NAME. Check $LOG_FILE." | tee -a $LOG_FILE
    fi

    # --- Step 5: Eval PTQ (Executorch) ---
    echo "\n--- Running 32_eval_ptq_executorch.sh ---" | tee -a $LOG_FILE
    docker compose run --rm -T spinquant bash -c "bash scripts/32_eval_ptq_executorch.sh $MODEL_NAME $W_BIT $A_BIT $KV_BIT $GROUPSIZE" >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR during 32_eval_ptq_executorch.sh for $MODEL_NAME. Check $LOG_FILE." | tee -a $LOG_FILE
    fi

    echo "\nCOMPLETED: $MODEL_NAME (W${W_BIT}A${A_BIT}KV${KV_BIT}GS${GROUPSIZE})" | tee -a $LOG_FILE

done <<< "$MODELS_TO_RUN"

echo "==========================================================================" | tee -a $LOG_FILE
echo "--- Batch Evaluation Finished ---" | tee -a $LOG_FILE