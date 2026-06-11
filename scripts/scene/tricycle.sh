declare -a splits=("train" "test" "val")
declare -a split_sizes=(300 150 150)
declare -a cycles=("orbit" "recolor" "resize" "linear" "rotate")

num_cycles=${#cycles[@]}

for ((split_idx=0; split_idx < ${#splits[@]}; split_idx++))
do
    split="${splits[split_idx]}"
    split_size="${split_sizes[split_idx]}"

    count=0
    sum_samples=0

    for ((i=0; i < num_cycles; i++))
    do
        echo "Split: $split, Size: $split_size, Cycles: 3x${cycles[i]}"
        count=$((count + 1))
        sum_samples=$((sum_samples + split_size))
        python -m cyclist.cyclist \
            --device CUDA \
            --split tricycle_"$split"_"${cycles[i]}"_"${cycles[i]}"_"${cycles[i]}" \
            --number_of_videos "$split_size" \
            --min_number_of_clutter_objects 2 \
            --max_number_of_clutter_objects 3 \
            --number_of_"${cycles[i]}"_cycles 3 \
            --force_generation

        for ((j=i+1; j < num_cycles; j++))
        do
            echo "Split: $split, Size: $split_size, Cycles: 2x${cycles[i]} 1x${cycles[j]}"
            count=$((count + 1))
            sum_samples=$((sum_samples + split_size))
            python -m cyclist.cyclist \
                --device CUDA \
                --split tricycle_"$split"_"${cycles[i]}"_"${cycles[i]}"_"${cycles[j]}" \
                --number_of_videos "$split_size" \
                --min_number_of_clutter_objects 2 \
                --max_number_of_clutter_objects 3 \
                --number_of_"${cycles[i]}"_cycles 2 \
                --number_of_"${cycles[j]}"_cycles 1 \
                --force_generation

            echo "Split: $split, Size: $split_size, Cycles: 1x${cycles[i]} 2x${cycles[j]}"
            count=$((count + 1))
            sum_samples=$((sum_samples + split_size))
            python -m cyclist.cyclist \
                --device CUDA \
                --split tricycle_"$split"_"${cycles[i]}"_"${cycles[j]}"_"${cycles[j]}" \
                --number_of_videos "$split_size" \
                --min_number_of_clutter_objects 2 \
                --max_number_of_clutter_objects 3 \
                --number_of_"${cycles[i]}"_cycles 1 \
                --number_of_"${cycles[j]}"_cycles 2 \
                --force_generation

            for ((k=j+1; k < num_cycles; k++))
            do
                echo "Split: $split, Size: $split_size, Cycles: 1x${cycles[i]} 1x${cycles[j]} 1x${cycles[k]}"
                count=$((count + 1))
                sum_samples=$((sum_samples + split_size))
                python -m cyclist.cyclist \
                    --device CUDA \
                    --split tricycle_"$split"_"${cycles[i]}"_"${cycles[j]}"_"${cycles[k]}" \
                    --number_of_videos "$split_size" \
                    --min_number_of_clutter_objects 2 \
                    --max_number_of_clutter_objects 3 \
                    --number_of_"${cycles[i]}"_cycles 1 \
                    --number_of_"${cycles[j]}"_cycles 1 \
                    --number_of_"${cycles[k]}"_cycles 1 \
                    --force_generation
            done
        done
    done
    echo "Total samples for $split: $sum_samples"
    echo "Total combinations: $count"
done
