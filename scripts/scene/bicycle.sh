declare -a splits=("train" "test" "validation")
declare -a split_sizes=(300 150 150)
declare -a cycles=("orbit" "recolor" "resize" "linear" "rotate")

num_cycles = ${#cycles[@]}

for split in "${splits[@]}"
do
    for split_size in ${split_sizes[@]}
    do
        for ((i=0; i < $num_cycles - 1; i++))
        do
            # Videos with twice the same cycle
            python -m cyclist.cyclist \
                --device CUDA \
                --split bicycle_"$split" \
                --number_of_videos $split_size \
                --min_number_of_clutter_objects 2 \
                --max_number_of_clutter_objects 3 \
                --number_of_"${cycles[i]}"_cycles 2 \
                --force_generation

            for ((j=i+1; j < $num_cycles; j++)) 
            do
                # Videos with combinations of two cycles
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
    done
done