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

    python main.py \
        --target "${target}" \
        --output_path "$output_path" \
        --output_folder "$output_folder" \
        --model "claude-3-5-sonnet-20240620" \
        --refine_iter 2
done
