import numpy as np
import pickle
import os
import sys
sys.path.append("..")
from utils import visualize_response_matrix

if __name__ == "__main__":
    # read long table
    with open(f"gsm_responses.pkl", "rb") as f:
        results_full = pickle.load(f)

    # keep useful columns, drop nan rows
    results_full = results_full.sample(frac=1).reset_index(drop=True)
    results = results_full[["request.model", "input.text", "references", "scenario", "benchmark", "dicho_score"]]
    results = results.dropna(subset=["request.model", "input.text", "references", "scenario", "benchmark", "dicho_score"])

    # drop the dicho_score of 0.5
    results = results[results["dicho_score"] != 0.5]
    results["dicho_score"] = results["dicho_score"].astype(bool)
    assert results["dicho_score"].isin([0, 1]).all()

    # drop duplicate rows
    results = results.drop_duplicates(subset=["request.model", "input.text", "references", "scenario", "benchmark"], keep='first')
    print(f"non-duplicate percentage:{results.shape[0]/results_full.shape[0]}")

    # pivot to turn long table into matrix
    results = results.pivot(index="request.model", columns=["input.text",  "references", "scenario", "benchmark"], values="dicho_score")
    # sort the columns by scenario
    results = results.sort_index(axis=1, level="scenario")

    # nan -> -1 -> np.nan
    results = results.fillna(-1).astype(int)
    results = results.replace(-1, np.nan)
    
    # Compute the overall average for each scenario manually
    scenario_means = {}
    for scenario in results.columns.get_level_values("scenario").unique():
        mask = results.columns.get_level_values("scenario") == scenario
        values = results.loc[:, mask].values  # all values for this scenario
        scenario_means[scenario] = np.nanmean(values)

    # Sort the scenario by their average score
    sorted_scenarios = sorted(scenario_means, key=scenario_means.get)

    # Create a mapping from scenario to its sort order
    scenario_order = {scenario: order for order, scenario in enumerate(sorted_scenarios)}

    # Reorder the columns based on the new scenario order using the key parameter
    results = results.sort_index(axis=1, level="scenario", key=lambda x: x.map(scenario_order))

    # Compute the overall average for each row (ignoring NaNs)
    row_means = results.mean(axis=1)

    # Sort the rows by these computed averages (lowest to highest)
    results = results.loc[row_means.sort_values().index]
    
    print(results.shape)
    print(f"missing percentage: {results.isna().values.sum() / (results.shape[0] * results.shape[1])}")
    
    # save
    with open("gsm_results.pkl", "wb") as f:
        pickle.dump(results, f)

    visualize_response_matrix(results, results, f"response_matrix")