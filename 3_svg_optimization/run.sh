#!/bin/bash
set -uo pipefail

gpu_id=0
experiment_name="example_generation"
output_size=256

targets=(
    "apple"
    "bonsai"
    "daisy"
    "ice_cream"
    "lighthouse"
    "penguin"
)

for target in "${targets[@]}"; do
    svg_folder="../output/${experiment_name}/${target}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python main.py \
        --svg_folder "${svg_folder}" \
        --target "${target}" \
        --output_size ${output_size} \
        --smoothness_weight_img 2.0 \
        --mse_loss_weight_img 2000.0 \
        --kl_weight_img 0.2 || {
        echo "Error processing target: ${target}"
        continue
    }
    
    echo "Finished processing target: ${target}"
done

echo "All targets processed"