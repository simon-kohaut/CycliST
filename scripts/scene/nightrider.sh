declare -a splits=("train" "test" "val")
declare -a split_sizes_uni=(64 32 32)
declare -a split_sizes_bi=(32 16 16)
declare -a split_sizes_tri=(16 8 8)
declare -a cycles=("orbit" "linear" "recolor" "resize" "rotate")

num_cycles=${#cycles[@]}

for ((split_idx=0; split_idx < ${#splits[@]}; split_idx++))
do
    split="${splits[split_idx]}"
    counter=0

    for cycle in "${cycles[@]}"
    do
        echo "Split: $split, Size: ${split_sizes_uni[split_idx]}, Cycles: 1x${cycle}"
        python -m cyclist.cyclist \
            --device CUDA \
            --split nightrider1_"$split"_"$cycle" \
            --number_of_videos ${split_sizes_uni[split_idx]} \
            --min_number_of_clutter_objects 2 \
            --max_number_of_clutter_objects 3 \
            --number_of_"$cycle"_cycles 1 \
            --force_generation \
            --cyclic_lights
        counter=$((counter + ${split_sizes_uni[split_idx]}))
    done

    for ((i=0; i < num_cycles; i++))
    do
        echo "Split: $split, Size: ${split_sizes_bi[split_idx]}, Cycles: 2x${cycles[i]}"
        python -m cyclist.cyclist \
            --device CUDA \
            --split nightrider2_"$split"_"${cycles[i]}"_"${cycles[i]}" \
            --number_of_videos ${split_sizes_bi[split_idx]} \
            --min_number_of_clutter_objects 2 \
            --max_number_of_clutter_objects 3 \
            --number_of_"${cycles[i]}"_cycles 2 \
            --force_generation \
            --cyclic_lights
        counter=$((counter + ${split_sizes_bi[split_idx]}))

        for ((j=i+1; j < num_cycles; j++))
        do
            echo "Split: $split, Size: ${split_sizes_bi[split_idx]}, Cycles: 1x${cycles[i]} 1x${cycles[j]}"
            python -m cyclist.cyclist \
                --device CUDA \
                --split nightrider2_"$split"_"${cycles[i]}"_"${cycles[j]}" \
                --number_of_videos ${split_sizes_bi[split_idx]} \
                --min_number_of_clutter_objects 2 \
                --max_number_of_clutter_objects 3 \
                --number_of_"${cycles[i]}"_cycles 1 \
                --number_of_"${cycles[j]}"_cycles 1 \
                --force_generation \
                --cyclic_lights
            counter=$((counter + ${split_sizes_bi[split_idx]}))
        done
    done

    for ((i=0; i < num_cycles; i++))
    do
        echo "Split: $split, Size: ${split_sizes_tri[split_idx]}, Cycles: 3x${cycles[i]}"
        python -m cyclist.cyclist \
            --device CUDA \
            --split nightrider3_"$split"_"${cycles[i]}"_"${cycles[i]}"_"${cycles[i]}" \
            --number_of_videos ${split_sizes_tri[split_idx]} \
            --min_number_of_clutter_objects 2 \
            --max_number_of_clutter_objects 3 \
            --number_of_"${cycles[i]}"_cycles 3 \
            --force_generation \
            --cyclic_lights
        counter=$((counter + ${split_sizes_tri[split_idx]}))

        for ((j=i+1; j < num_cycles; j++))
        do
            echo "Split: $split, Size: ${split_sizes_tri[split_idx]}, Cycles: 2x${cycles[i]} 1x${cycles[j]}"
            python -m cyclist.cyclist \
                --device CUDA \
                --split nightrider3_"$split"_"${cycles[i]}"_"${cycles[i]}"_"${cycles[j]}" \
                --number_of_videos ${split_sizes_tri[split_idx]} \
                --min_number_of_clutter_objects 2 \
                --max_number_of_clutter_objects 3 \
                --number_of_"${cycles[i]}"_cycles 2 \
                --number_of_"${cycles[j]}"_cycles 1 \
                --force_generation \
                --cyclic_lights
            counter=$((counter + ${split_sizes_tri[split_idx]}))

            echo "Split: $split, Size: ${split_sizes_tri[split_idx]}, Cycles: 1x${cycles[i]} 2x${cycles[j]}"
            python -m cyclist.cyclist \
                --device CUDA \
                --split nightrider3_"$split"_"${cycles[i]}"_"${cycles[j]}"_"${cycles[j]}" \
                --number_of_videos ${split_sizes_tri[split_idx]} \
                --min_number_of_clutter_objects 2 \
                --max_number_of_clutter_objects 3 \
                --number_of_"${cycles[i]}"_cycles 1 \
                --number_of_"${cycles[j]}"_cycles 2 \
                --force_generation \
                --cyclic_lights
            counter=$((counter + ${split_sizes_tri[split_idx]}))

            for ((k=j+1; k < num_cycles; k++))
            do
                echo "Split: $split, Size: ${split_sizes_tri[split_idx]}, Cycles: 1x${cycles[i]} 1x${cycles[j]} 1x${cycles[k]}"
                python -m cyclist.cyclist \
                    --device CUDA \
                    --split nightrider3_"$split"_"${cycles[i]}"_"${cycles[j]}"_"${cycles[k]}" \
                    --number_of_videos ${split_sizes_tri[split_idx]} \
                    --min_number_of_clutter_objects 2 \
                    --max_number_of_clutter_objects 3 \
                    --number_of_"${cycles[i]}"_cycles 1 \
                    --number_of_"${cycles[j]}"_cycles 1 \
                    --number_of_"${cycles[k]}"_cycles 1 \
                    --force_generation \
                    --cyclic_lights
                counter=$((counter + ${split_sizes_tri[split_idx]}))
            done
        done
    done
    echo "Total samples for $split: $counter"
done
