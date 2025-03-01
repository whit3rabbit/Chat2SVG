gpu_id=0

experiment_name="example_generation"
targets=(
    "apple"
    "bonsai"
    "daisy"
    "ice_cream"
    "lighthouse"
    "penguin"
)

for target in "${targets[@]}"; do
    output_path="../output"
    output_folder="${experiment_name}/${target}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python main.py \
        --target "${target}" \
        --output_path "$output_path" \
        --output_folder "$output_folder" \
        --seed 0 \
        --num_images_per_prompt 4 \
        --strength 1.0 \
        --thresh_iou 0.4
done
