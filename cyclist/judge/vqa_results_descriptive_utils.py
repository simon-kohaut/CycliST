import json
import pandas as pd
from pathlib import Path


MODEL_LATEX_MAPPING = {
    'OpenGVLab_InternVideo2_5_Chat_8B': r'\ivideo',
    'OpenGVLab_InternVL3-8B': r'\ivlsmall',
    'OpenGVLab_InternVL3-78B': r'\ivlbig',
    'lmms-lab_LLaVA-Video-7B-Qwen2': r'\llavavidsmall',
    'lmms-lab_LLaVA-Video-72B-Qwen2': r'\llavavidbig',
    'llava-hf_llava-onevision-qwen2-7b-ov-chat-hf': r'\llavaovsmall',
    'llava-hf_llava-onevision-qwen2-72b-ov-chat-hf': r'\llavaovbig',
    'gemini-2.0-flash': r'\geminiold',
    'gemini-2.5-flash': r'\gemininew',
}

MODEL_SHORT_NAMES = {
    'OpenGVLab_InternVideo2_5_Chat_8B': 'InternVideo2.5-8B',
    'OpenGVLab_InternVL3-8B': 'InternVL3-8B',
    'OpenGVLab_InternVL3-78B': 'InternVL3-78B',
    'lmms-lab_LLaVA-Video-7B-Qwen2': 'LLaVA-Video-7B',
    'lmms-lab_LLaVA-Video-72B-Qwen2': 'LLaVA-Video-72B',
    'llava-hf_llava-onevision-qwen2-7b-ov-chat-hf': 'LLaVA-OV-7B',
    'llava-hf_llava-onevision-qwen2-72b-ov-chat-hf': 'LLaVA-OV-72B',
    'gemini-2.0-flash': 'Gemini-2.0-Flash',
    'gemini-2.5-flash': 'Gemini-2.5-Flash',
}

_NIGHTRIDER_COMPARISONS = [
    ('nightrider1', 'unicycle',  'Nightrider1 vs Unicycle'),
    ('nightrider2', 'bicycle',   'Nightrider2 vs Bicycle'),
    ('nightrider3', 'tricycle',  'Nightrider3 vs Tricycle'),
]


def model_color(label):
    if 'InternVideo' in label:
        return '#9467bd'  # purple
    if 'InternVL' in label:
        return '#1f77b4'  # dark blue
    if 'LLaVA-Video' in label:
        return '#ff7f0e'
    if 'LLaVA-OV' in label:
        return '#2ca02c'
    if 'Gemini' in label:
        return '#d62728'
    return '#9467bd'


def model_marker(label):
    if '72B' in label or '78B' in label or '2.5' in label:
        return 's'   # square  → large models
    return 'o'       # circle  → small models (~7/8B) and Gemini 2.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sort_models(index, model_latex_mapping=None):
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING
    model_order = list(model_latex_mapping.keys())
    return sorted(index, key=lambda x: model_order.index(x) if x in model_order else len(model_order))


def _pivot_by_model_template(df):
    return df.pivot_table(
        index='model_clean', columns='template', values='accuracy', aggfunc='mean'
    ).round(1)


def _compute_nightrider_diff(results, nightrider_name, original_name):
    """Return (pivot_orig, pivot_night, diff_table) aligned to common models/templates."""
    df_orig = results[original_name].copy()
    df_night = results[nightrider_name].copy()
    for df in (df_orig, df_night):
        df['model_clean'] = df['model'].apply(lambda x: x.split('/')[-1])
    pivot_orig = _pivot_by_model_template(df_orig)
    pivot_night = _pivot_by_model_template(df_night)
    common_models = pivot_orig.index.intersection(pivot_night.index)
    common_templates = pivot_orig.columns.intersection(pivot_night.columns)
    pivot_orig = pivot_orig.loc[common_models, common_templates]
    pivot_night = pivot_night.loc[common_models, common_templates]
    return pivot_orig, pivot_night, pivot_night - pivot_orig


def _collect_templates_and_models(all_comparison_tables, model_latex_mapping=None):
    all_templates = sorted({t for d in all_comparison_tables.values()
                             for t in d['diff_table'].columns})
    all_models = _sort_models(
        {m for d in all_comparison_tables.values() for m in d['diff_table'].index},
        model_latex_mapping,
    )
    return all_templates, all_models


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_metrics(question_type='existential'):
    base_path = "output/eval/answers"

    print(f"Loading metrics for question type: {question_type}")

    templates = [
        f'questions_{question_type}_descriptive_attributes',
        f'questions_{question_type}_descriptive_relate',
        f'questions_{question_type}_descriptive_compare',
    ]

    models_strings = [
        'OpenGVLab_InternVideo2_5_Chat_8B',
        "OpenGVLab_InternVL3-8B",
        "OpenGVLab_InternVL3-78B",
        "lmms-lab_LLaVA-Video-7B-Qwen2",
        "lmms-lab_LLaVA-Video-72B-Qwen2",
        "llava-hf_llava-onevision-qwen2-7b-ov-chat-hf",
        "llava-hf_llava-onevision-qwen2-72b-ov-chat-hf",
        'gemini-2.0-flash',
        'gemini-2.5-flash',
    ]
    sample_frames = [32]
    datasets = ["unicycle", "bicycle", "tricycle", "unicycle_cluttered",
                "nightrider", "nightrider1", "nightrider2", "nightrider3"]
    splits = ["test"]

    dataset_results = {}

    for dataset in datasets:
        print(f"\nLoading metrics for dataset: {dataset}")
        dataset_metrics = []

        for sampled_frame in sample_frames:
            for split in splits:
                for template in templates:
                    for model in models_strings:
                        sf = 16 if model in (
                            "lmms-lab_LLaVA-Video-7B-Qwen2",
                            "lmms-lab_LLaVA-Video-72B-Qwen2",
                        ) else sampled_frame

                        metric_file = f"{template}_{dataset}_{sf}sf_metrics.json"
                        metrics_path = Path(base_path, dataset, split, template, model, metric_file)

                        if metrics_path.exists():
                            try:
                                with open(metrics_path, 'r') as f:
                                    metrics = json.load(f)
                                for key, value in metrics.items():
                                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                                        metrics[key] = round(float(value) * 100, 2)
                                dataset_metrics.append({
                                    'model': model,
                                    'template': template.replace(f'questions_{question_type}_descriptive_', ''),
                                    'question_type': question_type,
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
# Combined accuracy table
# ---------------------------------------------------------------------------

def generate_combined_latex_table(combined_table, model_latex_mapping=None):
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING
    columns = list(combined_table.columns)
    col_spec = 'l' + 'c' * len(columns)

    latex = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Combined Accuracy Results for All Datasets}\n"
        "\\label{tab:combined_accuracy}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
    )

    for index in _sort_models(combined_table.index, model_latex_mapping):
        row = combined_table.loc[index]
        cell_str = "".join(
            " & --" if (pd.isna(row[col]) or row[col] == 0) else f"&\\gradient{{{row[col]:.1f}}}"
            for col in columns
        )
        latex += f"&{model_latex_mapping.get(index, index)}{cell_str} \\\\\n"

    return latex


def create_combined_accuracy_table(results, datasets_to_use):
    if not results:
        print("No results to create tables from!")
        return None, None

    print("COMBINED ACCURACY TABLE - All Datasets")
    print("=" * 60)

    all_pivot_tables = []

    for dataset_name, df in results.items():
        if dataset_name not in datasets_to_use:
            continue
        if 'accuracy' not in df.columns:
            print(f"No accuracy column found for {dataset_name}")
            continue
        df_clean = df.copy()
        df_clean['model_clean'] = df_clean['model'].apply(lambda x: x.split('/')[-1])
        try:
            pivot = _pivot_by_model_template(df_clean).fillna(0)
            pivot.columns = [f"{dataset_name}_{col}" for col in pivot.columns]
            all_pivot_tables.append(pivot)
        except Exception as e:
            print(f"Error creating pivot table for {dataset_name}: {e}")

    if not all_pivot_tables:
        return None, None

    combined_table = pd.concat(all_pivot_tables, axis=1).fillna(0)
    print("\nCombined Table (Models vs Dataset_Template):")
    print("-" * 80)
    print(combined_table.to_string())

    latex_str = generate_combined_latex_table(combined_table)
    print(f"\nLaTeX table for combined results:\n{latex_str}")

    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    model_averages = combined_table.mean(axis=1)
    print(f"Best overall model: {model_averages.idxmax()} (avg: {model_averages.max():.1f}%)")

    for dataset in datasets_to_use:
        dataset_cols = [col for col in combined_table.columns if col.startswith(dataset)]
        if dataset_cols:
            dataset_avg = combined_table[dataset_cols].mean(axis=0)
            best_template = dataset_avg.idxmax().split('_')[-1]
            print(f"Best template for {dataset}: {best_template} (avg: {dataset_avg.max():.1f}%)")

    return combined_table, latex_str


# ---------------------------------------------------------------------------
# Nightrider effect table
# ---------------------------------------------------------------------------

def generate_nightrider_latex_table(diff_table, model_latex_mapping=None):
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING
    columns = list(diff_table.columns)
    col_spec = 'l' + 'c' * len(columns)

    header = "Model" + "".join(f" & {col.title()}" for col in columns) + " \\\\\n"
    latex = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Accuracy Drop: Nightrider vs Unicycle (Percentage Points)}\n"
        "\\label{tab:nightrider_effect}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
        + header + "\\hline\n"
    )

    for model in _sort_models(diff_table.index, model_latex_mapping):
        cell_str = "".join(
            " & --" if pd.isna(diff_table.loc[model, col])
            else f"&\\gradientb{{{diff_table.loc[model, col]:+.1f}}}"
            for col in columns
        )
        latex += f"&{model_latex_mapping.get(model, model)}{cell_str} \\\\\n"

    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex


def create_nightrider_effect_table(results):
    required = ['unicycle', 'bicycle', 'tricycle', 'nightrider1']
    if not all(ds in results for ds in required):
        print(f"Need all of {required} results!")
        return None, None

    print("\nNIGHTRIDER EFFECT TABLE\n" + "=" * 80)

    pivot_orig, pivot_night, diff_table = _compute_nightrider_diff(results, 'nightrider', 'unicycle')

    combined_rows = [
        {
            'Model': model, 'Template': template,
            'Avg(Uni)': f"{pivot_orig.loc[model, template]:.1f}",
            'Nightrider': f"{pivot_night.loc[model, template]:.1f}",
            'Difference': f"{diff_table.loc[model, template]:+.1f}",
        }
        for model in pivot_orig.index
        for template in pivot_orig.columns
    ]
    print("\nDetailed Comparison:")
    print(pd.DataFrame(combined_rows).to_string(index=False))
    print("\nDifference Only (Nightrider - Unicycle):")
    print(diff_table.to_string())
    print(f"\nAverage accuracy drop: {diff_table.values.mean():.1f}%")
    print(f"Worst affected template: {diff_table.mean(axis=0).idxmin()} ({diff_table.mean(axis=0).min():.1f}%)")
    print(f"Most robust model: {diff_table.mean(axis=1).idxmax()} ({diff_table.mean(axis=1).max():.1f}%)")

    latex_str = generate_nightrider_latex_table(diff_table)
    print(f"\nLATEX TABLE:\n{latex_str}")

    return pd.DataFrame(combined_rows), diff_table


# ---------------------------------------------------------------------------
# Nightrider split comparisons
# ---------------------------------------------------------------------------

def generate_individual_nightrider_latex_diff(diff_table, nightrider_name, original_name,
                                              model_latex_mapping=None):
    """One column per template, showing only the accuracy difference."""
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING
    templates = list(diff_table.columns)
    orig_title = original_name.title()
    night_title = nightrider_name.replace('nightrider', 'NR')

    header = "Model" + "".join(f" & {t.title()}" for t in templates) + " \\\\\n"
    latex = (
        "\\begin{table}[h!]\n\\centering\n"
        f"\\caption{{Accuracy Drop: {orig_title} vs {night_title.upper()} (Percentage Points)}}\n"
        f"\\label{{tab:{nightrider_name}_vs_{original_name}}}\n"
        f"\\begin{{tabular}}{{{'l' + 'c' * len(templates)}}}\n\\hline\n"
        + header + "\\hline\n"
    )

    for model in _sort_models(diff_table.index, model_latex_mapping):
        cell_str = "".join(
            f"&\\gradientb{{{diff_table.loc[model, t]:+.1f}}}"
            if t in diff_table.columns and model in diff_table.index
            else " & --"
            for t in templates
        )
        latex += f"&{model_latex_mapping.get(model, model)}{cell_str} \\\\\n"

    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex


def generate_individual_nightrider_latex_full(pivot_orig, pivot_night, diff_table,
                                              nightrider_name, original_name,
                                              model_latex_mapping=None):
    """Three columns per template: Orig / NR / Δ."""
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING
    templates = list(diff_table.columns)
    orig_title = original_name.title()
    night_title = nightrider_name.replace('nightrider', 'NR')

    header1 = "".join(f" & \\multicolumn{{3}}{{c}}{{{t.title()}}}" for t in templates) + " \\\\\n"
    clines = "".join(f"\\cline{{{2 + i * 3}-{3 + i * 3}}}" for i in range(len(templates)))
    header2 = "Model" + "".join(f" & {orig_title[:4]} & {night_title[:4]} & Δ"
                                 for _ in templates) + " \\\\\n"
    latex = (
        "\\begin{table}[h!]\n\\centering\n"
        f"\\caption{{{orig_title} vs {night_title.upper()} Comparison}}\n"
        f"\\label{{tab:{nightrider_name}_vs_{original_name}}}\n"
        f"\\begin{{tabular}}{{{'l' + 'ccc' * len(templates)}}}\n\\hline\n"
        + header1 + clines + "\n" + header2 + "\\hline\n"
    )

    for model in _sort_models(diff_table.index, model_latex_mapping):
        cell_str = "".join(
            (f"&\\gradient{{{pivot_orig.loc[model, t]:.1f}}}"
             f"&\\gradient{{{pivot_night.loc[model, t]:.1f}}}"
             f"&\\gradientb{{{diff_table.loc[model, t]:+.1f}}}")
            if t in pivot_orig.columns and model in pivot_orig.index
            else " & -- & -- & --"
            for t in templates
        )
        latex += f"&{model_latex_mapping.get(model, model)}{cell_str} \\\\\n"

    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex


def generate_combined_nightrider_latex(all_comparison_tables, comparisons=None,
                                       model_latex_mapping=None):
    if comparisons is None:
        comparisons = _NIGHTRIDER_COMPARISONS
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING

    all_templates, all_models = _collect_templates_and_models(all_comparison_tables, model_latex_mapping)

    latex_tables = []
    for template in all_templates:
        header2 = "Model" + " & Orig & NR & Δ" * len(comparisons) + " \\\\\n"
        latex = (
            "\\begin{table}[h!]\n\\centering\n"
            f"\\caption{{Nightrider Effect on {template.title()} Template}}\n"
            f"\\label{{tab:nightrider_{template}}}\n"
            f"\\begin{{tabular}}{{{'l' + 'ccc' * len(comparisons)}}}\n\\hline\n"
            "& \\multicolumn{3}{c}{Unicycle/NR1}"
            " & \\multicolumn{3}{c}{Bicycle/NR2}"
            " & \\multicolumn{3}{c}{Tricycle/NR3} \\\\\n"
            "\\cline{2-10}\n"
            + header2 + "\\hline\n"
        )

        for model in all_models:
            cell_str = ""
            for _, _, comparison_title in comparisons:
                if comparison_title not in all_comparison_tables:
                    cell_str += " & -- & -- & --"
                    continue
                d = all_comparison_tables[comparison_title]
                if template in d['pivot_orig'].columns and model in d['pivot_orig'].index:
                    cell_str += (f"&\\gradient{{{d['pivot_orig'].loc[model, template]:.1f}}}"
                                 f"&\\gradient{{{d['pivot_night'].loc[model, template]:.1f}}}"
                                 f"&\\gradientb{{{d['diff_table'].loc[model, template]:+.1f}}}")
                else:
                    cell_str += " & -- & -- & --"
            latex += f"&{model_latex_mapping.get(model, model)}{cell_str} \\\\\n"

        latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
        latex_tables.append(latex)

    return "\n\n".join(latex_tables)


def generate_all_nightriders_combined_latex(all_comparison_tables, comparisons=None,
                                            model_latex_mapping=None):
    if comparisons is None:
        comparisons = _NIGHTRIDER_COMPARISONS
    if model_latex_mapping is None:
        model_latex_mapping = MODEL_LATEX_MAPPING

    all_templates, all_models = _collect_templates_and_models(all_comparison_tables, model_latex_mapping)

    header1 = "".join(
        f" & \\multicolumn{{{len(all_templates)}}}{{c}}{{{nr.replace('nightrider', 'NR').upper()}}}"
        for nr, _, _ in comparisons
    ) + " \\\\\n"
    clines = "".join(
        f"\\cline{{{2 + i * len(all_templates)}-{1 + (i + 1) * len(all_templates)}}}"
        for i in range(len(comparisons))
    )
    header2 = "Model" + "".join(
        f" & {t.title()[:3]}" for _ in comparisons for t in all_templates
    ) + " \\\\\n"

    latex = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Accuracy Drop: All Nightrider Splits Across Templates (Percentage Points)}\n"
        "\\label{tab:all_nightrider_combined}\n"
        f"\\begin{{tabular}}{{{'l' + 'c' * (len(comparisons) * len(all_templates))}}}\n\\hline\n"
        + header1 + clines + "\n" + header2 + "\\hline\n"
    )

    for model in all_models:
        cell_str = ""
        for _, _, comparison_title in comparisons:
            for t in all_templates:
                if comparison_title not in all_comparison_tables:
                    cell_str += " & --"
                    continue
                d = all_comparison_tables[comparison_title]
                cell_str += (
                    f"&\\gradientb{{{d['diff_table'].loc[model, t]:+.1f}}}"
                    if t in d['diff_table'].columns and model in d['diff_table'].index
                    else " & --"
                )
        latex += f"&{model_latex_mapping.get(model, model)}{cell_str} \\\\\n"

    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex


def compare_all_nightrider_splits(results, comparisons=None):
    if comparisons is None:
        comparisons = _NIGHTRIDER_COMPARISONS

    all_comparison_tables = {}

    for nightrider_name, original_name, comparison_title in comparisons:
        if nightrider_name not in results or original_name not in results:
            print(f"Missing data for {comparison_title}")
            continue

        print(f"\n{'=' * 80}\n{comparison_title}\n{'=' * 80}")
        pivot_orig, pivot_night, diff_table = _compute_nightrider_diff(
            results, nightrider_name, original_name
        )
        print(f"Found {len(pivot_orig.index)} common models")
        print(f"Found {len(pivot_orig.columns)} common templates: {list(pivot_orig.columns)}")
        print(f"Average accuracy drop: {diff_table.values.mean():.1f}%")

        all_comparison_tables[comparison_title] = {
            'diff_table': diff_table,
            'pivot_orig': pivot_orig,
            'pivot_night': pivot_night,
        }

    latex_str = generate_combined_nightrider_latex(all_comparison_tables, comparisons)
    individual_latex_tables = [
        generate_individual_nightrider_latex_diff(
            all_comparison_tables[title]['diff_table'], nr, orig
        )
        for nr, orig, title in comparisons
        if title in all_comparison_tables
    ]

    return all_comparison_tables, latex_str, individual_latex_tables


def create_individual_nightrider_tables(results, comparisons=None):
    if comparisons is None:
        comparisons = _NIGHTRIDER_COMPARISONS

    all_latex_tables = []

    for nightrider_name, original_name, comparison_title in comparisons:
        if nightrider_name not in results or original_name not in results:
            print(f"Missing data for {comparison_title}")
            continue

        print(f"\n{'=' * 80}\n{comparison_title}\n{'=' * 80}")
        pivot_orig, pivot_night, diff_table = _compute_nightrider_diff(
            results, nightrider_name, original_name
        )

        print(f"Found {len(pivot_orig.index)} common models")
        for t in pivot_orig.columns:
            print(f"\n{t.upper()}")
            print(f"{'Model':<40} {'Original':>10} {'Nightrider':>10} {'Difference':>10}")
            print("-" * 75)
            for model in _sort_models(pivot_orig.index):
                print(f"{model:<40} {pivot_orig.loc[model, t]:>10.1f}"
                      f" {pivot_night.loc[model, t]:>10.1f}"
                      f" {diff_table.loc[model, t]:>+10.1f}")

        print(f"\nAverage accuracy drop: {diff_table.values.mean():.1f}%")

        latex_str = generate_individual_nightrider_latex_full(
            pivot_orig, pivot_night, diff_table, nightrider_name, original_name
        )
        all_latex_tables.append(latex_str)
        print(f"\nLaTeX Table for {comparison_title}:\n{latex_str}")

    return all_latex_tables


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def build_wide(results, template_filter=None, tiers=('unicycle', 'bicycle', 'tricycle')):
    """Return a DataFrame: index=model short name, columns=tiers, values=mean accuracy."""
    rows = []
    for tier in tiers:
        if tier not in results:
            continue
        df_tier = results[tier].copy()
        df_tier['model_clean'] = df_tier['model'].apply(lambda x: x.split('/')[-1])
        if template_filter is not None:
            df_tier = df_tier[df_tier['template'].str.endswith(template_filter)]
        rows.append(df_tier.groupby('model_clean')['accuracy'].mean().rename(tier))
    df = pd.concat(rows, axis=1).dropna()
    df.index = df.index.map(lambda x: MODEL_SHORT_NAMES.get(x, x))
    return df
