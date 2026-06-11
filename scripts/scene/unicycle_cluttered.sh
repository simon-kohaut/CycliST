declare -a splits=("train" "test" "val")
declare -a split_sizes=(300 150 150)
declare -a cycles=("orbit" "recolor" "resize" "linear" "rotate")

for ((split_idx=0; split_idx < ${#splits[@]}; split_idx++))
do
    split="${splits[split_idx]}"
    split_size="${split_sizes[split_idx]}"

    for cycle in "${cycles[@]}"
    do
        echo "Split: $split, Size: $split_size, Cycles: 1x${cycle}"
        python -m cyclist.cyclist \
            --device CUDA \
            --split unicycle_cluttered_"$split" \
            --number_of_videos $split_size \
            --min_number_of_clutter_objects 4 \
            --max_number_of_clutter_objects 9 \
            --number_of_"$cycle"_cycles 1 \
            --force_generation
    done
done
