import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
from pathlib import Path

from query_sglang import query_judge_batched
from metrics import compute_count_metrics, compute_bool_metrics, compute_attribute_metrics

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "assets" / "prompts"

with open(_PROMPTS_DIR / "vqa_prompt_boolean.txt") as f:
    boolean_judge_prompt = f.read()

with open(_PROMPTS_DIR / "vqa_prompt_counting.txt") as f:
    counting_judge_prompt = f.read()

with open(_PROMPTS_DIR / "vqa_prompt_attribute.txt") as f:
    attribute_judge_prompt = f.read()


# Path where eval_vqa.py wrote its answers (set --answer_path in eval_vqa.sh)
base_path = "output/eval/answers"

templates_desc_existential = [
    'questions_existential_descriptive_attributes',
    'questions_existential_descriptive_relate',
    'questions_existential_descriptive_compare',
]
templates_desc_universal = [
    'questions_universal_descriptive_attributes',
    'questions_universal_descriptive_relate',
    'questions_universal_descriptive_compare',
]
templates_cyclic = [
    'questions_cycle_representative_clockwise',
    'questions_cycle_representative_orbit',
    'questions_cycle_representative_transition',
]
templates_numeric = [
    'questions_counting_cycles',
    'questions_counting_frequency',
    'questions_counting_occurence',
]
templates = templates_desc_existential + templates_desc_universal + templates_cyclic + templates_numeric

gemini_models = ["gemini-2.5-flash", "gemini-2.0-flash"]
intern_video_models = ['OpenGVLab_InternVideo2_5_Chat_8B']
intern_vl_models = ["OpenGVLab_InternVL3-8B", "OpenGVLab_InternVL3-78B"]
ov_models = ["llava-hf_llava-onevision-qwen2-7b-ov-chat-hf", "llava-hf_llava-onevision-qwen2-72b-ov-chat-hf"]
lvid_models = ["lmms-lab_LLaVA-Video-7B-Qwen2", "lmms-lab_LLaVA-Video-72B-Qwen2"]
all_models = gemini_models + intern_video_models + intern_vl_models + ov_models + lvid_models

models_strings = all_models
datasets = ["unicycle"]
splits = ["test"]
sample_frames = [32]

for sample_frame in sample_frames:
    for dataset in datasets:
        for split in splits:
            for template in templates:

                if "descriptive" in template:
                    judge_prompt = boolean_judge_prompt
                elif "representative" in template:
                    judge_prompt = attribute_judge_prompt
                elif "counting" in template:
                    judge_prompt = counting_judge_prompt
                else:
                    raise ValueError(f"No judge configured for template: {template}")

                for ms in models_strings:
                    sf = 16 if ms in lvid_models else sample_frame
                    print(f"Processing {template} - {ms} ({sf}sf)")

                    try:
                        file_name = f"{template}_{dataset}_{sf}sf.csv"
                        metric_file = f"{template}_{dataset}_{sf}sf_metrics.json"
                        answer_csv = Path(base_path, dataset, split, template, ms, file_name)
                        metrics_json = Path(base_path, dataset, split, template, ms, metric_file)

                        results, _ = query_judge_batched(answer_csv, judge_prompt)

                        if "descriptive" in template:
                            metrics, _ = compute_bool_metrics(results)
                        elif "representative" in template:
                            metrics, _ = compute_attribute_metrics(results)
                        elif "counting" in template:
                            metrics, _ = compute_count_metrics(results)

                        with open(metrics_json, 'w') as f:
                            json.dump(metrics, f, indent=4)
                        print(metrics)

                    except Exception as e:
                        print(e)
