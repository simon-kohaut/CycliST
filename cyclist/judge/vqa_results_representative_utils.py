import json
import pandas as pd
from pathlib import Path

from vqa_results_descriptive_utils import MODEL_LATEX_MAPPING, _sort_models

BASE_PATH = "output/eval/answers"

_LLAVA_16F_MODELS = {
    "lmms-lab_LLaVA-Video-7B-Qwen2",
    "lmms-lab_LLaVA-Video-72B-Qwen2",
}

_DEFAULT_MODELS = [

    'lmms-lab_LLaVA-Video-7B-Qwen2',
    'lmms-lab_LLaVA-Video-72B-Qwen2',
    'llava-hf_llava-onevision-qwen2-7b-ov-chat-hf',
    'llava-hf_llava-onevision-qwen2-72b-ov-chat-hf',
    'gemini-2.5-flash',
    'gemini-2.0-flash',
    'OpenGVLab_InternVL3-8B',
    'OpenGVLab_InternVL3-78B',
    'OpenGVLab_InternVideo2_5_Chat_8B',

]

_TEMPLATES = [
    'questions_cycle_representative_clockwise',
    'questions_cycle_representative_orbit',
    'questions_cycle_representative_transition',
]

_DATASETS = ['unicycle', 'bicycle', 'tricycle', 'unicycle_cluttered', 'nightrider']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_metrics():
    dataset_results = {}

    for dataset in _DATASETS:
        print("-" * 10)
        print(f"Loading metrics for dataset: {dataset}")
        dataset_metrics = []

        for sf_default in [32]:
            for split in ['test']:
                for template in _TEMPLATES:
                    for model in _DEFAULT_MODELS:
                        sf = 16 if model in _LLAVA_16F_MODELS else sf_default

                        metric_file = f"{template}_{dataset}_{sf}sf_metrics.json"
                        metrics_path = Path(BASE_PATH, dataset, split, template, model, metric_file)

                        if metrics_path.exists():
                            try:
                                with open(metrics_path, 'r') as f:
                                    metrics = json.load(f)

                                for key, value in metrics.items():
                                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                                        metrics[key] = round(float(value) * 100, 2)

                                dataset_metrics.append({
                                    'model': model,
                                    'template': template.split('_')[-1],
                                    'sample_frames': sf,
                                    **metrics,
                                })
                                print(f"✓ Loaded: {model} - {template.split('_')[-1]}")
                            except Exception as e:
                                print(f"✗ Error loading {metrics_path}: {e}")
                        else:
                            print(f"✗ File not found: {metrics_path}")

        if dataset_metrics:
            df = pd.DataFrame(dataset_metrics)
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = df[numeric_cols].round(2)
            dataset_results[dataset] = df
        else:
            print(f"No metrics files found for dataset: {dataset}")

    return dataset_results


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def create_pivot_table(df, dataset_name=None):
    df = df.copy()
    df['model_clean'] = df['model'].apply(lambda x: x.split('/')[-1])
    pivot = df.pivot_table(
        index='model_clean', columns='template', values='accuracy', aggfunc='mean'
    ).round(1)
    if dataset_name:
        pivot.columns = [f"{dataset_name}_{c}" for c in pivot.columns]
    return pivot


def print_summary_stats(df, title):
    print(f"\n{title}")
    print("-" * 40)
    if df.ndim == 2:
        model_avg = df.mean(axis=1)
        print(f"Best model: {model_avg.idxmax()} ({model_avg.max():.1f}%)")
        template_avg = df.mean(axis=0)
        print(f"Best template: {template_avg.idxmax()} ({template_avg.max():.1f}%)")
        print(f"Overall average: {df.values.mean():.1f}%)")


def generate_latex_table(df, caption, label, value_formatter=None, grad_str="gradient"):
    columns  = list(df.columns)
    col_spec = 'l' + 'c' * len(columns)

    latex  = f"\\begin{{table}}[h!]\n"
    latex += f"\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += f"\\hline\n"
    latex += "Model" + "".join(f" & {c.title()}" for c in columns) + " \\\\\n"
    latex += "\\hline\n"

    for index in _sort_models(df.index):
        row = df.loc[index]
        model_latex = MODEL_LATEX_MAPPING.get(index, index)
        row_str = model_latex
        for col in columns:
            value = row[col]
            if pd.isna(value) or value == 0:
                row_str += " & --"
            elif value_formatter:
                row_str += f" & {value_formatter(value)}"
            else:
                row_str += f" & \\{grad_str}{{{value:.1f}}}"
        row_str += " \\\\\n"
        latex += row_str

    latex += "\\hline\n\\end{tabular}\n\\end{table}"
    return latex


# ---------------------------------------------------------------------------
# Template analysis
# ---------------------------------------------------------------------------

def show_template_results(results, template_name, grad_str="gradient"):
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {template_name.upper()} TEMPLATE")
    print(f"{'='*60}\n")

    filtered = {
        ds: df[df['template'] == template_name].copy()
        for ds, df in results.items()
        if not df[df['template'] == template_name].empty
    }

    if not filtered:
        print(f"No results found for template: {template_name}")
        return None

    combined = pd.concat(
        [create_pivot_table(df, ds) for ds, df in filtered.items()], axis=1
    ).fillna(0)

    print(combined.to_string())
    print_summary_stats(combined, f"SUMMARY - {template_name}")

    latex = generate_latex_table(
        combined,
        f"Accuracy Results - {template_name.title()} Template",
        f"tab:accuracy_{template_name}",
        grad_str=grad_str,
    )
    print(f"\n{latex}")
    return combined
