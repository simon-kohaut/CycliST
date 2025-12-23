declare -a splits=("train" "test" "validation")
declare -a split_sizes=(300 150 150)
declare -a cycles=("orbit" "recolor" "resize" "linear" "rotate")

num_cycles = ${#cycles[@]}

for split in "${splits[@]}"
do
    for split_size in ${split_sizes[@]}
    do
        for ((i=0; i < $num_cycles - 2; i++))
        do
            # Videos with trice the same cycle
            python -m cyclist.cyclist \
                --device CUDA \
                --split tricycle_"$split" \
                --number_of_videos $split_size \
                --min_number_of_clutter_objects 2 \
                --max_number_of_clutter_objects 3 \
                --number_of_"${cycles[i]}"_cycles 3 \
                --force_generation

            for ((j=i+1; j < $num_cycles - 1; j++)) 
            do
                 # Videos with combinations of two plus one cycles
                python -m cyclist.cyclist \
                    --device CUDA \
                    --split tricycle_"$split" \
                    --number_of_videos $split_size \
                    --min_number_of_clutter_objects 2 \
                    --max_number_of_clutter_objects 3 \
                    --number_of_"${cycles[i]}"_cycles 2 \
                    --number_of_"${cycles[j]}"_cycles 1 \
                    --force_generation

                python -m cyclist.cyclist \
                    --device CUDA \
                    --split tricycle_"$split" \
                    --number_of_videos $split_size \
                    --min_number_of_clutter_objects 2 \
                    --max_number_of_clutter_objects 3 \
                    --number_of_"${cycles[i]}"_cycles 1 \
                    --number_of_"${cycles[j]}"_cycles 2 \
                    --force_generation

                for ((k=j+1; k < $num_cycles; k++)) 
                do
                    # Videos with combinations of three different cycles
                    python -m cyclist.cyclist \
                        --device CUDA \
                        --split tricycle_"$split" \
                        --number_of_videos $split_size \
                        --min_number_of_clutter_objects 2 \
                        --max_number_of_clutter_objects 3 \
                        --number_of_"${cycles[i]}"_cycles 1 \
                        --number_of_"${cycles[j]}"_cycles 1 \
                        --number_of_"${cycles[k]}"_cycles 1 \
                        --force_generation
                done
            done
        done
    done
done
