import argparse
import json
import os

import matplotlib.patheffects as PathEffects
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_spatial_relations(json_path: str, output_dir: str):
    """
    Generates and saves a plot of spatial relations over time from a scene
    configuration JSON file.

    The plot will show which spatial relations (e.g., 'left', 'right')
    are active at each time frame.

    Args:
        json_path: Path to the input JSON configuration file.
        output_dir: Directory to save the output plot.
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, "r") as f:
        config = json.load(f)

    if "relationships" not in config:
        print(f"Error: 'relationships' key not found in {json_path}")
        return

    if "objects" not in config:
        print(f"Error: 'objects' key not found in {json_path}")
        return

    objects = config["objects"]

    def get_obj_desc(obj_index):
        obj = objects[obj_index]
        return f"{obj.get('color', '')} {obj.get('mesh', 'object')}".strip()

    # A dictionary to hold combined relations for each pair at each frame
    # Format: {(frame, target_idx, source_idx): [relations...]}
    combined_relations = {}

    # First, gather all relations for each pair at each frame
    for relation, frames in config["relationships"].items():
        if relation in ["above", "below"]:
            continue
        for frame, pairs in frames.items():
            for target_idx, source_idx in pairs:
                key = (int(frame), target_idx, source_idx)
                if key not in combined_relations:
                    combined_relations[key] = []
                combined_relations[key].append(relation)

    # Now, create the plot data with combined relation strings
    plot_data = []
    for (frame, target_idx, source_idx), relations in combined_relations.items():
        # Sort to ensure consistent naming (e.g., 'front left', not 'left front')
        relations.sort()
        combined_relation_str = " ".join(relations)

        source_desc = get_obj_desc(source_idx)
        target_desc = get_obj_desc(target_idx)

        # To avoid mirrored plots (e.g., o1 vs o2 and o2 vs o1), we
        # enforce a canonical order for the pair. We'll use the object
        # with the lexicographically smaller description as the "source".
        if source_desc > target_desc:
            continue  # Skip this pair to avoid duplication

        pair_group = f"'{source_desc}' relative to '{target_desc}'"
        plot_data.append(
            {
                "time": frame + 1,
                "relation": combined_relation_str,
                "pair_group": pair_group,
            }
        )

    if not plot_data:
        print("No relationship data to plot.")
        return

    df = pd.DataFrame(plot_data)

    # Define the desired cyclic order for the y-axis relations.
    # This ensures the y-axis is sorted logically, not alphabetically.
    relation_order = [
        "left",
        "front left",
        "front",
        "front right",
        "right",
        "behind right",
        "behind",
        "behind left",
    ]
    df['relation'] = pd.Categorical(df['relation'], categories=relation_order, ordered=True)

    df = df.sort_values(by=["pair_group", "time"])

    # Determine the range for x-ticks in steps of 32
    max_frame = df['time'].max()
    # Start with ticks at multiples of 32
    xticks_32 = np.arange(32, max_frame + 1, 32)
    xticks = np.insert(xticks_32, 0, 1)  # Add a tick for the first frame

    # Use relplot to create subplots for each object pair.
    # This creates a FacetGrid, and we can customize it after creation.
    g = sns.relplot(
        data=df,
        x="time",
        y="relation",
        col="pair_group",
        col_wrap=5,
        kind="line",
        height=3,
        aspect=1.2,
        linewidth=2.5,
        facet_kws={'sharey': True, 'sharex': True},
    )

    g.set_axis_labels("Frame", "", fontsize=18)
    g.set_titles("")  # Clear default titles to set custom ones
    g.set(xticks=xticks)
    g.tick_params(axis='both', which='major', labelsize=18)

    # Manually set titles with colored text for each subplot
    for pair_group_str, ax in g.axes_dict.items():
        # Parse the pair_group string like "'blue Cube' relative to 'red Cube'"
        parts = pair_group_str.replace("'", "").split(" relative to ")
        source_full_desc, target_full_desc = parts[0], parts[1]

        source_color, source_mesh = source_full_desc.split(" ", 1)
        target_color, target_mesh = target_full_desc.split(" ", 1)

        # Use a darker yellow for better readability on a white background
        if source_color == "yellow":
            source_color = "goldenrod"
        if target_color == "yellow":
            target_color = "goldenrod"

        # Set the title with colored object names
        ax.set_title("")  # Clear the base text
        text_effects = [PathEffects.withStroke(linewidth=1.5, foreground='black')]
        ax.text(
            0.3,
            1.02,
            f"{source_mesh}",
            color=source_color,
            ha='right',
            va='bottom',
            transform=ax.transAxes,
            # weight='bold',
            # path_effects=text_effects,
            fontsize=20
        )
        ax.text(0.5,  1.02, " is ... of ", color='black', ha='center', va='bottom', transform=ax.transAxes, fontsize=20)
        ax.text(
            0.7,
            1.02,
            f"{target_mesh}",
            color=target_color,
            ha='left',
            va='bottom',
            transform=ax.transAxes,
            # weight='bold',
            # path_effects=text_effects,
            fontsize=20
        )

    # # Adjust subplot parameters to prevent title overlap and fix alignment
    # # This is a more robust alternative to tight_layout() for this use case.
    # g.fig.subplots_adjust(top=0.9, hspace=0.4, wspace=0.2)
    g.fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_path = os.path.join(output_dir, f"relations.pdf")

    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot spatial relations from a JSON file.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the scene config JSON file.")
    parser.add_argument("--output_dir", type=str, default="../output/plots", help="Directory to save the plot.")
    args = parser.parse_args()
    plot_spatial_relations(args.json_path, args.output_dir)