import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tueplots import bundles
import matplotlib.colors as mcolors
from datasets import load_dataset
import os
import argparse

def custom_sort_key(x):
    suffix = x.split("-")[-1]
    if suffix == "Chat":
        return float('inf') - 1  # Second last
    elif suffix in ["Safe", "Instruct"]:
        return float('inf')      # Last
    else:
        return int(suffix)  # Regular numeric sorting

def visualize_response_matrix(results, value, filename):
    # Extract the groups labels in the order of the columns
    group_values = results.columns.get_level_values("scenario")

    # Identify the boundaries where the group changes
    boundaries = []
    for i in range(1, len(group_values)):
        if group_values[i] != group_values[i - 1]:
            boundaries.append(i - 0.5)  # using 0.5 to place the line between columns

    # Visualize the results with a matrix: red is 0, white is -1 and blue is 1
    cmap = mcolors.ListedColormap(["white", "red", "blue"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Calculate midpoints for each group label
    groups_list = list(group_values)
    group_names = []
    group_midpoints = []
    current_group = groups_list[0]
    start_index = 0
    for i, grp in enumerate(groups_list):
        if grp != current_group:
            midpoint = (start_index + i - 1) / 2.0
            group_names.append(current_group)
            group_midpoints.append(midpoint)
            current_group = grp
            start_index = i
    # Add the last group
    midpoint = (start_index + len(groups_list) - 1) / 2.0
    group_names.append(current_group)
    group_midpoints.append(midpoint)

    # Define the minimum spacing between labels (e.g., 100 units)
    min_spacing = 100
    last_label_pos = -float("inf")
    # Plot the matrix
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(20, 10))
        cax = ax.matshow(value, aspect="auto", cmap=cmap, norm=norm)

        # Add vertical lines at each boundary
        for b in boundaries:
            ax.axvline(x=b, color="black", linewidth=0.25, linestyle="--", alpha=0.5)
        
        # Add group labels above the matrix, only if they're spaced enough apart
        for name, pos in zip(group_names, group_midpoints):
            if pos - last_label_pos >= min_spacing:
                ax.text(pos, -5, name, ha='center', va='bottom', rotation=90, fontsize=3)
                last_label_pos = pos

        # Add model labels on the y-axis
        ax.set_yticks(range(len(results.index)))
        ax.set_yticklabels(results.index, fontsize=3)

        # Add a colorbar
        cbar = plt.colorbar(cax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["-1", "0", "1"])
        plt.savefig(f"../result/{filename}.png", dpi=600, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    # EleutherAI/pythia-6.9b, EleutherAI/pythia-12b
    # LLM360/Amber, allenai/OLMo-2-0325-32B, HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
    args = parser.parse_args()
    model_name = args.repo_id.split("/")[1]
    
    # Load your pre-generated responses pickle file.
    with open(f"../data/gather_ckpt_data/responses_{model_name}.pkl", "rb") as f:
        results_full = pickle.load(f)

    results_full = results_full.sample(frac=1).reset_index(drop=True)
    results = results_full[["request.model", "request.prompt", "scenario", "dicho_score"]]
    results = results.dropna(subset=["request.model", "request.prompt", "scenario", "dicho_score"])
    
    # Drop the dicho_score of 0.5
    results = results[results["dicho_score"] != 0.5]
    results["dicho_score"] = results["dicho_score"].astype(bool)
    assert results["dicho_score"].isin([0, 1]).all()
    
    # drop duplicate
    results = results.drop_duplicates(subset=["request.model", "request.prompt", "scenario"], keep='first')
    print(f"duplicate percentage: {results.shape[0]/results_full.shape[0]}")

    # Pivot the DataFrame so that rows are models and columns are a MultiIndex of (request.prompt, scenario)
    results = results.pivot(index="request.model", columns=["request.prompt", "scenario"], values="dicho_score")
    
    # Reindex the DataFrame according to the step order
    breakpoint()
    sorted_index = sorted(results.index, key=custom_sort_key)
    results = results.reindex(sorted_index)
    
    # Sort the columns by scenario groups
    results = results.sort_index(axis=1, level="scenario")

    # # Remove columns that are all 0 or all 1 and fill missing values with -1 temporarily
    # TODO: remove 0&nan, and 1&nan
    # results = results.loc[:, (results != 0).any()]
    # results = results.loc[:, (results != 1).any()]
    
    # nan -> -1 -> np.nan
    results = results.fillna(-1).astype(int)
    # Replace -1 with NaN so that missing scores are ignored during visualization
    results = results.replace(-1, np.nan)

    # Compute the overall average for each scenario group manually
    group_means = {}
    for group in results.columns.get_level_values("scenario").unique():
        mask = results.columns.get_level_values("scenario") == group
        values = results.loc[:, mask].values  # all values for this group
        group_means[group] = np.nanmean(values)

    # Sort the scenario groups by their average score
    sorted_groups = sorted(group_means, key=group_means.get)
    group_order = {group: order for order, group in enumerate(sorted_groups)}

    # Reorder the columns based on the new group order
    results = results.sort_index(axis=1, level="scenario", key=lambda x: x.map(group_order))

    # print(f"missing percentage: {np.isnan(results).sum() / (results.shape[0] * results.shape[1])}")

    output_dir = "../result/gather_ckpt_data"
    os.makedirs(output_dir, exist_ok=True)
    visualize_response_matrix(results, results, f"{output_dir}/response_matrix_{model_name}")
    
    # Load all splits from the dataset
    dataset = load_dataset("stair-lab/reeval-difficulty-for-helm")

    # Create a dictionary mapping request.prompt -> z
    prompt_to_z = {}
    for split in dataset.keys():
        for example in dataset[split]:
            prompt = example.get("request.prompt")
            z_value = example.get("z")
            prompt_to_z[prompt] = z_value

    new_columns = []
    for col in results.columns:
        # In our current MultiIndex, level 0 is "request.prompt" and level 1 is "scenario"
        prompt = col[0]
        z_val = prompt_to_z.get(prompt, np.nan)
        new_columns.append((prompt, z_val, col[1]))
    
    # Set the new MultiIndex with three levels: "request.prompt", "z", and "scenario"
    results.columns = pd.MultiIndex.from_tuples(new_columns, names=["request.prompt", "z", "scenario"])
    
    # Save the final results with the new column level
    with open(f"../data/gather_ckpt_data/results_{model_name}.pkl", "wb") as f:
        pickle.dump(results, f)
