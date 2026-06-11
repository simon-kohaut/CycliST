#!/bin/bash

base_path="$(cd "$(dirname "$0")/../.." && pwd)"

MODEL_PARAM=1
DATASET_PARAM="unicycle"
FPS_PARAM=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)   MODEL_PARAM="$2";   shift 2 ;;
        --dataset) DATASET_PARAM="$2"; shift 2 ;;
        --fps)     FPS_PARAM="$2";     shift 2 ;;
        *)
            echo "Usage: $0 [--model 1-9] [--dataset unicycle|bicycle|tricycle|unicycle_cluttered|nightrider] [--fps <number>]"
            exit 1 ;;
    esac
done

case $MODEL_PARAM in
    1) SELECTED_MODEL="lmms-lab/LLaVA-Video-7B-Qwen2" ;;
    2) SELECTED_MODEL="llava-hf/llava-onevision-qwen2-7b-ov-chat-hf" ;;
    3) SELECTED_MODEL="OpenGVLab/InternVideo2_5_Chat_8B" ;;
    4) SELECTED_MODEL="gemini-2.0-flash" ;;
    5) SELECTED_MODEL="gemini-2.5-flash" ;;
    6) SELECTED_MODEL="lmms-lab/LLaVA-Video-72B-Qwen2" ;;
    7) SELECTED_MODEL="llava-hf/llava-onevision-qwen2-72b-ov-chat-hf" ;;
    8) SELECTED_MODEL="OpenGVLab/InternVL3-8B" ;;
    9) SELECTED_MODEL="OpenGVLab/InternVL3-78B" ;;
    *) echo "Invalid model parameter (1-9)"; exit 1 ;;
esac

echo "Model: $SELECTED_MODEL | Dataset: $DATASET_PARAM | FPS: $FPS_PARAM"

universal_templates=("universal_descriptive_attributes" "universal_descriptive_compare" "universal_descriptive_relate")
existential_templates=("existential_descriptive_attributes" "existential_descriptive_compare" "existential_descriptive_relate")
cyclic_representative=("cycle_representative_orbit" "cycle_representative_clockwise" "cycle_representative_transition")
counting=("counting_cycles" "counting_frequency" "counting_occurence")

template_selection=("${universal_templates[@]}" "${existential_templates[@]}" "${cyclic_representative[@]}" "${counting[@]}")
# Single-template override for debugging:
# template_selection=("counting_cycles")

for split in "test"
do
    for template in "${template_selection[@]}"
    do
        DATA_PATH="${base_path}/output/videos/${DATASET_PARAM}/${split}"
        QUESTION_FILE="${base_path}/output/questions/${DATASET_PARAM}/${split}/questions_${template}.json"
        ANSWER_PATH="${base_path}/output/eval/answers/${DATASET_PARAM}/${split}/${template}"
        EXP_NAME="${template}_${DATASET_PARAM}_${FPS_PARAM}sf"

        echo "Template: $template"
        python3 "$base_path/cyclist/eval/eval_vqa.py" \
            --SAMPLED_FRAMES_PER_SEC $FPS_PARAM \
            --model_id "$SELECTED_MODEL" \
            --data_path "$DATA_PATH" \
            --answer_path "$ANSWER_PATH" \
            --question_file "$QUESTION_FILE" \
            --experiment_name "$EXP_NAME"
    done
done
