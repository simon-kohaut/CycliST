import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
from pathlib import Path

from cyclist.judge.query_sglang import query_judge_su_batched
from cyclist.judge.metrics import compute_match_metrics

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "assets" / "prompts"

with open(_PROMPTS_DIR / "scene_understanding_map_prompt.txt") as f:
    scene_judge_prompt = f.read()


# Path where eval_scene_understanding.py wrote its captions (set --captions_path in eval_scene_understanding.sh)
base_path = "output/eval/scene_understanding"

gemini_models = ["gemini-2.5-flash", "gemini-2.0-flash"]
intern_video_models = ['OpenGVLab_InternVideo2_5_Chat_8B']
intern_vl_models = ["OpenGVLab_InternVL3-8B", "OpenGVLab_InternVL3-78B"]
ov_models = ["llava-hf_llava-onevision-qwen2-7b-ov-chat-hf", "llava-hf_llava-onevision-qwen2-72b-ov-chat-hf"]
lvid_models = ["lmms-lab_LLaVA-Video-7B-Qwen2", "lmms-lab_LLaVA-Video-72B-Qwen2"]
all_models = gemini_models + intern_video_models + intern_vl_models + ov_models + lvid_models

models_strings = all_models
datasets = ["unicycle", "bicycle", "tricycle"]
splits = ["test"]
sample_frames = [32]

for sample_frame in sample_frames:
    for dataset in datasets:
        for split in splits:
            for ms in models_strings:
                sf = 16 if ms in lvid_models else sample_frame
                print(f"Processing {dataset} - {ms} ({sf}sf)")

                try:
                    file_name = f"scene_understanding_{dataset}_{sf}sf.csv"
                    metric_file = f"scene_understanding_{dataset}_{sf}sf_metrics.json"
                    answer_csv = Path(base_path, dataset, split, ms, file_name)
                    metrics_json = Path(base_path, dataset, split, ms, metric_file)

                    if not answer_csv.exists():
                        print(f"Answers CSV not found, skipping: {answer_csv}")
                        continue

                    results, _, obj_lists, responses = query_judge_su_batched(answer_csv, scene_judge_prompt)
                    metrics = compute_match_metrics(results, obj_lists, responses)

                    with open(metrics_json, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    print(metrics)

                except Exception as e:
                    print(e)

print("Done")
