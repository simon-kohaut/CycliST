declare -a splits=("train" "test" "val")
declare -a split_sizes=(300 150 150)
declare -a cycles=("orbit" "recolor" "resize" "linear" "rotate")

num_cycles=${#cycles[@]}
total_count=0

for ((split_idx=0; split_idx < ${#splits[@]}; split_idx++))
do
    split="${splits[split_idx]}"
    split_size="${split_sizes[split_idx]}"

    split_samples=0
    for ((i=0; i < num_cycles; i++))
    do
        echo "Split: $split, Size: $split_size, Cycles: 2x${cycles[i]}"
        split_samples=$((split_samples + split_size))
        total_count=$((total_count + 1))
        python -m cyclist.cyclist \
            --device CUDA \
            --split bicycle_"$split" \
            --number_of_videos $split_size \
            --min_number_of_clutter_objects 2 \
            --max_number_of_clutter_objects 3 \
            --number_of_"${cycles[i]}"_cycles 2 \
            --force_generation

        for ((j=i+1; j < num_cycles; j++))
        do
            echo "Split: $split, Size: $split_size, Cycles: 1x${cycles[i]} 1x${cycles[j]}"
            split_samples=$((split_samples + split_size))
            total_count=$((total_count + 1))
            python -m cyclist.cyclist \
                --device CUDA \
                --split bicycle_"$split" \
                --number_of_videos $split_size \
                --min_number_of_clutter_objects 2 \
                --max_number_of_clutter_objects 3 \
                --number_of_"${cycles[i]}"_cycles 1 \
                --number_of_"${cycles[j]}"_cycles 1 \
                --force_generation
        done
    done
    echo "Total samples for $split: $split_samples"
done

echo "Total combinations: $total_count"
