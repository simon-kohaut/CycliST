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
    'OpenGVLab_InternVideo2_5_Chat_8B',
    'OpenGVLab_InternVL3-8B',
    'OpenGVLab_InternVL3-78B',
    'lmms-lab_LLaVA-Video-7B-Qwen2',
    'lmms-lab_LLaVA-Video-72B-Qwen2',
    'llava-hf_llava-onevision-qwen2-7b-ov-chat-hf',
    'llava-hf_llava-onevision-qwen2-72b-ov-chat-hf',
    'gemini-2.0-flash',
    'gemini-2.5-flash',
]

_DEFAULT_DATASETS = ['unicycle', 'bicycle', 'tricycle']
_DEFAULT_TEMPLATES = [
    'questions_counting_cycles',
    'questions_counting_frequency',
    'questions_counting_occurence',
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_metrics(templates=None, verbose=True):
    if templates is None:
        templates = _DEFAULT_TEMPLATES

    dataset_results = {}

    for dataset in _DEFAULT_DATASETS:
        if verbose:
            print(f"\nLoading metrics for dataset: {dataset}")
        dataset_metrics = []

        for sf_default in [32]:
            for split in ['test']:
                for template in templates:
                    for model in _DEFAULT_MODELS:
                        sf = 16 if model in _LLAVA_16F_MODELS else sf_default

                        metric_file = f"{template}_{dataset}_{sf}sf_metrics.json"
                        metrics_path = Path(BASE_PATH, dataset, split, template, model, metric_file)

                        if metrics_path.exists():
                            try:
                                with open(metrics_path, 'r') as f:
                                    metrics = json.load(f)

                                processed = {}
                                for key, value in metrics.items():
                                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                                        if key == 'accuracy':
                                            processed[key] = round(float(value) * 100, 2)
                                        elif key == 'avg_abs_error':
                                            processed[key] = round(float(value), 4)
                                        else:
                                            processed[key] = round(float(value) * 100, 2)
                                    else:
                                        processed[key] = value

                                dataset_metrics.append({
                                    'model': model,
                                    'template': template.replace('questions_', ''),
                                    'sample_frames': sf,
                                    **processed,
                                })
                                if verbose:
                                    print(f"✓ Loaded: {model} - {template.split('_')[-1]}")
                            except Exception as e:
                                if verbose:
                                    print(f"✗ Error loading {metrics_path}: {e}")
                        else:
                            if verbose:
                                print(f"✗ File not found: {metrics_path}")

        if dataset_metrics:
            df = pd.DataFrame(dataset_metrics)
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = df[numeric_cols].round(2)
            dataset_results[dataset] = df
        elif verbose:
            print(f"No metrics files found for dataset: {dataset}")

    return dataset_results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_pivot(df, value='accuracy'):
    df = df.copy()
    df['model_clean'] = df['model'].apply(lambda x: x.split('/')[-1])
    return df.pivot_table(
        index='model_clean', columns='template', values=value, aggfunc='mean'
    ).round(1 if value == 'accuracy' else 4)


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def create_combined_accuracy_table(results, grad_str_acc="gradient", grad_str_err="gradientb"):
    if not results:
        print("No results to create tables from!")
        return None, None, None, None

    print("COMBINED ACCURACY TABLE - All Datasets")
    print("=" * 60)

    all_acc_pivots = []
    all_err_pivots = []

    for dataset_name, df in results.items():
        if 'accuracy' not in df.columns:
            print(f"No accuracy column found for {dataset_name}")
            continue

        try:
            pivot_acc = _build_pivot(df, 'accuracy').fillna(0)
            pivot_acc.columns = [f"{dataset_name}_{c}" for c in pivot_acc.columns]
            all_acc_pivots.append(pivot_acc)
        except Exception as e:
            print(f"Error creating accuracy pivot for {dataset_name}: {e}")

        if 'avg_abs_error' in df.columns:
            try:
                pivot_err = _build_pivot(df, 'avg_abs_error').fillna(0)
                pivot_err.columns = [f"{dataset_name}_{c}" for c in pivot_err.columns]
                all_err_pivots.append(pivot_err)
            except Exception as e:
                print(f"Error creating error pivot for {dataset_name}: {e}")

    if not all_acc_pivots:
        return None, None, None, None

    combined_table = pd.concat(all_acc_pivots, axis=1).fillna(0)

    print("\nCombined Table (Models vs Dataset_Template) - ACCURACY:")
    print("-" * 80)
    print(combined_table.to_string())

    latex_str = generate_combined_latex_table(combined_table, metric_name="Accuracy", grad_str=grad_str_acc)
    print(f"\nLaTeX table for combined accuracy results:\n{latex_str}")

    _print_accuracy_summary(combined_table)

    combined_error_table, latex_error_str = None, None

    if all_err_pivots:
        combined_error_table = pd.concat(all_err_pivots, axis=1).fillna(0)

        print("\n" + "=" * 80)
        print("COMBINED AVG ABSOLUTE ERROR TABLE - All Datasets")
        print("=" * 80)
        print("\nCombined Table (Models vs Dataset_Template) - AVG ABS ERROR:")
        print("-" * 80)
        print(combined_error_table.to_string())

        latex_error_str = generate_combined_latex_table(
            combined_error_table, metric_name="Average Absolute Error", grad_str=grad_str_err
        )
        print(f"\nLaTeX table for combined avg abs error results:\n{latex_error_str}")

        _print_error_summary(combined_error_table)

    return combined_table, latex_str, combined_error_table, latex_error_str


def _print_accuracy_summary(combined_table):
    print("\nSUMMARY STATISTICS - ACCURACY:")
    print("-" * 40)
    model_avgs = combined_table.mean(axis=1)
    print(f"Best overall model: {model_avgs.idxmax()} (avg: {model_avgs.max():.1f}%)")
    for dataset in ['unicycle', 'bicycle', 'tricycle', 'unicycle_cluttered']:
        cols = [c for c in combined_table.columns if c.startswith(dataset)]
        if cols:
            avg = combined_table[cols].mean(axis=0)
            print(f"Best template for {dataset}: {avg.idxmax().split('_')[-1]} (avg: {avg.max():.1f}%)")


def _print_error_summary(combined_error_table):
    print("\nSUMMARY STATISTICS - AVG ABS ERROR:")
    print("-" * 40)
    model_avgs = combined_error_table.mean(axis=1)
    print(f"Best overall model (lowest error): {model_avgs.idxmin()} (avg error: {model_avgs.min():.4f})")
    for dataset in ['unicycle', 'bicycle', 'tricycle', 'unicycle_cluttered']:
        cols = [c for c in combined_error_table.columns if c.startswith(dataset)]
        if cols:
            avg = combined_error_table[cols].mean(axis=0)
            print(f"Best template for {dataset}: {avg.idxmin().split('_')[-1]} (avg error: {avg.min():.4f})")


def generate_combined_latex_table(combined_table, model_latex_mapping=None, metric_name="Accuracy", grad_str="gradient"):
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING

    columns = list(combined_table.columns)
    col_spec = 'l' + 'c' * len(columns)
    is_error = "Error" in metric_name

    latex = f"\\begin{{table}}[h!]\n"
    latex += f"\\centering\n"
    latex += f"\\caption{{Combined {metric_name} Results for All Datasets}}\n"
    latex += f"\\label{{tab:combined_{metric_name.lower().replace(' ', '_')}}}\n"
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += f"\\hline\n"

    for index in _sort_models(combined_table.index, model_latex_mapping):
        row = combined_table.loc[index]
        model_latex = model_latex_mapping.get(index, index)
        row_str = f"&{model_latex}"
        for col in columns:
            value = row[col]
            if pd.isna(value) or value == 0:
                row_str += " & --"
            elif is_error:
                row_str += f"&\\{grad_str}{{{value:.2f}}}"
            else:
                row_str += f"&\\{grad_str}{{{value:.1f}}}"
        row_str += " \\\\\n"
        latex += row_str

    return latex


def create_combined_metrics_table(combined_table, combined_error_table, grad_str_acc="gradient", grad_str_err="gradientb"):
    if combined_table is None or combined_error_table is None:
        print("No data to create table!")
        return None

    print("\nCOMBINED METRICS TABLE - Accuracy & Avg Absolute Error")
    print("=" * 120)

    datasets = list(dict.fromkeys(c.split('_')[0] for c in combined_table.columns))
    models = combined_table.index.tolist()

    table_rows = []
    for model in models:
        row = {'Model': model}
        for dataset in datasets:
            acc_cols = [c for c in combined_table.columns if c.startswith(dataset)]
            val = combined_table.loc[model, acc_cols[0]] if acc_cols else None
            row[f"{dataset.upper()}\nAccuracy (%)"] = f"{val:.1f}" if val and val > 0 else "--"
        for dataset in datasets:
            err_cols = [c for c in combined_error_table.columns if c.startswith(dataset)]
            val = combined_error_table.loc[model, err_cols[0]] if err_cols else None
            row[f"{dataset.upper()}\nAvg Abs Error"] = f"{val:.2f}" if val and val > 0 else "--"
        table_rows.append(row)

    combined_df = pd.DataFrame(table_rows)
    print("\nTable Format: Models (rows) × Dataset Metrics (columns)")
    print("-" * 120)
    print(combined_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("LATEX TABLE - Combined Metrics (Models as rows)")
    print("=" * 120)
    latex_str = generate_combined_metrics_latex(
        combined_table, combined_error_table, datasets, models, grad_str_acc, grad_str_err
    )
    print(latex_str)

    return combined_df


def generate_combined_metrics_latex(combined_table, combined_error_table, datasets, models,
                                     grad_str_acc="gradient", grad_str_err="gradientb",
                                     model_latex_mapping=None):
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING

    col_spec = 'l' + 'c' * len(datasets) + 'c' * len(datasets)

    latex = "\\begin{table}[h!]\n"
    latex += "\\centering\n"
    latex += "\\caption{Combined Accuracy and Average Absolute Error Results by Model}\n"
    latex += "\\label{tab:combined_metrics_models}\n"
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += "\\hline\n"

    latex += (f"Model & \\multicolumn{{{len(datasets)}}}{{c}}{{Accuracy (\\%)}}"
              f" & \\multicolumn{{{len(datasets)}}}{{c}}{{Avg Abs Error}} \\\\\n")
    latex += "".join(f" & {d.upper()}" for d in datasets) * 2 + " \\\\\n"
    latex += "\\hline\n"

    for model in _sort_models(models, model_latex_mapping):
        model_latex = model_latex_mapping.get(model, model)
        row_str = model_latex
        for dataset in datasets:
            acc_cols = [c for c in combined_table.columns if c.startswith(dataset)]
            val = combined_table.loc[model, acc_cols[0]] if acc_cols else 0
            row_str += f" & \\{grad_str_acc}{{{val:.1f}}}" if val > 0 else " & --"
        for dataset in datasets:
            err_cols = [c for c in combined_error_table.columns if c.startswith(dataset)]
            val = combined_error_table.loc[model, err_cols[0]] if err_cols else 0
            row_str += f" & \\{grad_str_err}{{{val:.2f}}}" if val > 0 else " & --"
        row_str += " \\\\\n"
        latex += row_str

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


# ---------------------------------------------------------------------------
# Clutter effect
# ---------------------------------------------------------------------------

def create_clutter_effect_table(results, grad_str="gradientb"):
    if 'unicycle' not in results or 'unicycle_cluttered' not in results:
        print("Need both unicycle and unicycle_cluttered results!")
        return None, None

    print("\nCLUTTER EFFECT TABLE - Accuracy Drop from Unicycle to Unicycle_Cluttered")
    print("=" * 80)

    pivot_uni  = _build_pivot(results['unicycle'],          'accuracy')
    pivot_clut = _build_pivot(results['unicycle_cluttered'], 'accuracy')
    diff_table = pivot_clut - pivot_uni

    combined_rows = [
        {
            'Model': model, 'Template': template,
            'Unicycle':   f"{pivot_uni.loc[model, template]:.1f}",
            'Cluttered':  f"{pivot_clut.loc[model, template]:.1f}",
            'Difference': f"{diff_table.loc[model, template]:+.1f}",
        }
        for model    in pivot_uni.index
        for template in pivot_uni.columns
    ]

    combined_df = pd.DataFrame(combined_rows)
    print("\nDetailed Comparison:")
    print(combined_df.to_string(index=False))
    print("\n\nDifference Only (Cluttered - Unicycle):")
    print("-" * 60)
    print(diff_table.to_string())

    print("\n\nSUMMARY STATISTICS:")
    print("-" * 40)
    print(f"Average accuracy drop across all: {diff_table.values.mean():.1f}%")
    print(f"Worst affected template: {diff_table.mean(axis=0).idxmin()} ({diff_table.mean(axis=0).min():.1f}%)")
    print(f"Most robust model: {diff_table.mean(axis=1).idxmax()} ({diff_table.mean(axis=1).max():.1f}%)")
    print(f"Least robust model: {diff_table.mean(axis=1).idxmin()} ({diff_table.mean(axis=1).min():.1f}%)")

    latex_str = generate_clutter_latex_table(diff_table, grad_str=grad_str)
    print(f"\n\nLATEX TABLE - Accuracy Difference (Cluttered - Unicycle):")
    print("=" * 80)
    print(latex_str)

    return combined_df, diff_table


def generate_clutter_latex_table(diff_table, model_latex_mapping=None, grad_str="gradientb"):
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING

    columns  = list(diff_table.columns)
    col_spec = 'l' + 'c' * len(columns)

    latex  = "\\begin{table}[h!]\n"
    latex += "\\centering\n"
    latex += "\\caption{Accuracy Drop: Unicycle Cluttered vs Unicycle (Percentage Points)}\n"
    latex += "\\label{tab:clutter_effect}\n"
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += "\\hline\n"
    latex += "Model" + "".join(f" & {c.title()}" for c in columns) + " \\\\\n"
    latex += "\\hline\n"

    for model in _sort_models(diff_table.index, model_latex_mapping):
        model_latex = model_latex_mapping.get(model, model)
        row_str = model_latex
        for col in columns:
            value = diff_table.loc[model, col]
            row_str += " & --" if pd.isna(value) else f"&\\{grad_str}{{{value:+.1f}}}"
        row_str += " \\\\\n"
        latex += row_str

    return latex
