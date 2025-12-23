#!/bin/bash

# Default parameters
SAMPLED_FRAMES_PER_SEC=16
# MODEL_ID="lmms-lab/LLaVA-Video-7B-Qwen2"
# MODEL_ID="llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
# MODEL_ID='OpenGVLab/InternVideo2_5_Chat_8B'

# for dataset in  "unicycle"  "bicycle" "tricycle" "unicycle_cluttered";
# for dataset in  "bicycle" "tricycle" "unicycle";
for dataset in "tricycle";
do

    DATA_PATH="/pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist/output/videos/${dataset}/test"
    QUESTION_FILE="/pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist/output/questions/universal_descriptive_count/universal_descriptive_count_${dataset}.json"
    ANSWER_PATH="/pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist/output/eval/answers/"
    VERBOSE=false
    EXP_NAME="universal_descriptive_count_${dataset}_${SAMPLED_FRAMES_PER_SEC}fps"


    # iterate over all model IDS
    for MODEL_ID in "lmms-lab/LLaVA-Video-7B-Qwen2" "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf" "OpenGVLab/InternVideo2_5_Chat_8B"; 
    # for MODEL_ID in "gemini-2.0-flash";
    do
        echo "Running evaluation for model: $MODEL_ID"
        python3 /pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist/cyclist/eval/eval_vqa.py \
            --SAMPLED_FRAMES_PER_SEC $SAMPLED_FRAMES_PER_SEC \
            --model_id $MODEL_ID \
            --data_path $DATA_PATH \
            --answer_path $ANSWER_PATH \
            --question_file $QUESTION_FILE \
            --experiment_name $EXP_NAME \
            ${VERBOSE:+--verbose}
    done
done