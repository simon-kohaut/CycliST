#!/bin/bash

# Default parameters
SAMPLED_FRAMES_PER_SEC=8
# MODEL_ID="lmms-lab/LLaVA-Video-7B-Qwen2"
# MODEL_ID="llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
# MODEL_ID='OpenGVLab/InternVideo2_5_Chat_8B'

# for dataset in  "unicycle"  "bicycle" "tricycle" "unicycle_cluttered";
for dataset in  "bicycle" "tricycle" "unicycle";
do

    DATA_PATH="/pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist/output/videos/${dataset}/test"
    QUESTION_FILE="/pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist/output/questions/cycle_frequency/cycle_frequency_${dataset}.json"
    ANSWER_PATH="/pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist/output/eval/"
    VERBOSE=false
    EXP_NAME="cycle_frequency_${dataset}"


    # iterate over all model IDS
    for MODEL_ID in "lmms-lab/LLaVA-Video-72B-Qwen2" "llava-hf/llava-onevision-qwen2-72b-ov-chat-hf" ;
    # for MODEL_ID in "gemini-2.0-flash" "OpenGVLab/InternVideo2_5_Chat_8B"; 
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