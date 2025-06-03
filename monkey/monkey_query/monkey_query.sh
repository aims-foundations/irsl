#!/usr/bin/env bash

# List of datasets to process sequentially
datasets=(math med_qa commonsense)

for ds in "${datasets[@]}"; do
    echo "Running monkey_query on dataset: $ds"
    CUDA_VISIBLE_DEVICES=6 \
    python monkey_query.py \
        --dataset "$ds" \
        --model_nickname mistralai/Mistral-7B-v0.1 \
        --num_samples_per_sampling_call 256

    if [ $? -ne 0 ]; then
        echo "Error encountered on dataset $ds. Exiting."
        exit 1
    fi
done

echo "All jobs completed."
