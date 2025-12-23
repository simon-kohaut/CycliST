declare -a splits=("train" "test" "validation")
declare -a split_sizes=(300 150 150)
declare -a cycles=("orbit" "recolor" "resize" "linear" "rotate")

for split in "${splits[@]}"
do
    for split_size in ${split_sizes[@]}
    do
        for cycle in "${cycles[@]}"
        do
            python -m cyclist.cyclist \
                --device CUDA \
                --split unicycle_"$split" \
                --number_of_videos $split_size \
                --min_number_of_clutter_objects 2 \
                --max_number_of_clutter_objects 3 \
                --number_of_"$cycle"_cycles 1 \
                --force_generation
        done
    done
done