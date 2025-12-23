import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.load_env_vars import load_env

load_env()

from torch.utils.data import DataLoader
import json 
import re

from tqdm import tqdm

from judge.query_sglang import query_judge_batched 
from judge.metrics import compute_count_metrics, compute_bool_metrics
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# load txt file

with open('/pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist_dev/Cyclist/assets/prompts/vqa_prompt_boolean.txt', 'r') as f:
    lines = f.readlines()
    judge_prompt = ''.join(lines)
    print(judge_prompt)
    
    
base_path= "/pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist_dev/Cyclist/output/eval/answers"

templates= ['questions_existential_descriptive_attributes', 'questions_existential_descriptive_relate', 'questions_existential_descriptive_compare']
models_strings=["llava-hf_llava-onevision-qwen2-7b-ov-chat-hf", "lmms-lab_LLaVA-Video-7B-Qwen2", 'OpenGVLab_InternVideo2_5_Chat_8B']

# for fps in [2,4,8,16]:
for sample_frames in [16]:
    for dataset in ["unicycle"]:
        for split in ["test"]:
            for template in templates:
                for ms in models_strings:
                            
                    file_name = f"{template}_{dataset}_{sample_frames}sf.csv"
                    metric_file = f"{template}_{dataset}_{sample_frames}sf_metrics.json"
                    
                    answer_csv = Path(base_path, dataset, split, template,ms, file_name)
                    metrics_json = Path(base_path, dataset, split, template,ms, metric_file)

                    print("Evaluating", answer_csv)
                    
                    results, requests = query_judge_batched(answer_csv, judge_prompt)
                    metrics, cm = compute_bool_metrics(results)
                    
                    # save metrics to json file
                    with open(metrics_json, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    print(metrics)

                    # save confusion matrix to json file
                        #save confusion matrix to file

                    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    # disp.plot(cmap=plt.cm.Blues)
                    # title= model_path.split("/")[-1]
                    # plt.title(f"{title} on {dataset}")
                    # #make sure the confusion directory exists
                    # Path(model_path, "confusion").mkdir(parents=True, exist_ok=True)
                    # plt.savefig(Path(model_path,"confusion", metric_file.replace(".csv", "_cm.png")))
                    # plt.show()
        
# /pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist_dev/Cyclist/output/eval/answers/unicycle/test/questions_existential_descriptive_attributes/llava-hf_llava-onevision-qwen2-7b-ov-chat-hf/questions_existential_descriptive_attributes_unicycle_16sf.csv
