import numpy as np
import pandas as pd
import pickle
from datasets import load_dataset
import os
import argparse
import sys
sys.path.append("..")
from utils import visualize_response_matrix

def custom_sort_key(x):
    suffix = x.split("-")[-1]
    if suffix == "Chat":
        return float('inf') - 1  # Second last
    elif suffix in ["Safe", "Instruct"]:
        return float('inf')      # Last
    else:
        return int(suffix)  # Regular numeric sorting

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
    sorted_index = sorted(results.index, key=custom_sort_key)
    results = results.reindex(sorted_index)
    
    # Sort the columns by scenario groups
    results = results.sort_index(axis=1, level="scenario")

    # # Remove columns that are all 0 or all 1 and fill missing values with -1 temporarily
    # results = results.loc[:, ~((results.isin([0, np.nan]).all()) | (results.isin([1, np.nan]).all()))]

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

    print(f"missing percentage: {results.isna().values.sum() / (results.shape[0] * results.shape[1])}")

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
