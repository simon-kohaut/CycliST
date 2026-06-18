#!/bin/bash

base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PARAM=1
DATASET_PARAM="unicycle"
FPS_PARAM=32

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

for split in "test"
do
    DATA_PATH="${base_path}/output/videos/${DATASET_PARAM}/${split}"
    SCENE_PATH="${base_path}/output/scenes/${DATASET_PARAM}/${split}"
    CAPTIONS_PATH="${base_path}/output/eval/scene_understanding/${DATASET_PARAM}/${split}"
    EXP_NAME="scene_understanding_${DATASET_PARAM}_${FPS_PARAM}sf"

    python3 -m cyclist.eval.eval_scene_understanding \
        --SAMPLED_FRAMES_PER_SEC $FPS_PARAM \
        --model_id "$SELECTED_MODEL" \
        --data_path "$DATA_PATH" \
        --scene_path "$SCENE_PATH" \
        --captions_path "$CAPTIONS_PATH" \
        --experiment_name "$EXP_NAME"
done
