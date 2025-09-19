import os
import pickle
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tueplots import bundles
from monkey_generalize_scaleup import scenario2benchmark
import pandas as pd
import seaborn as sns

model_name_map = {
    'meta-llama-Meta-Llama-3-8B-Instruct': 'Meta-Llama-3-8B-Instruct',
    'meta-llama-Meta-Llama-3-70B': 'Meta-Llama-3-70B-Instruct',
    'pythia-12b': 'Pythia_12B',
    'pythia-6.9b': 'Pythia_6.9B',
}

# 1) find all files
dir = "../../result/monkey_generalize_scaleup_diff_split"
pattern = f"{dir}/monkey_scaleup_data_*.pkl"
filepaths = glob.glob(pattern)
scenarios = scenario2benchmark.keys()

# 2) group results by scenario
results_by_scenario = {}
for fp in filepaths:
    fname = os.path.basename(fp)
    # get the core: "monkey_scaleup_data_{model}_{scenario}.pkl" → "{model}_{scenario}"
    core = fname.removeprefix("monkey_scaleup_data_").removesuffix(".pkl")
    scenario = next((s for s in scenarios if core.endswith(f"_{s}")), None)
    if scenario == "harm_bench":
        continue
    model_name = core[: -(len(scenario) + 1)]
    if model_name.startswith("Pythia") and not (
        model_name.endswith("6.9B") or model_name.endswith("12B")
    ):
        continue
    model_name = model_name_map[model_name] if model_name in model_name_map.keys() else model_name
    
    # load and compute MSEs
    with open(fp, "rb") as f:
        res = pickle.load(f)
    gt        = np.exp(-res["test_pass_datk_gts"])
    pred_dist = np.exp(-res["train_pass_datk_est2s"])
    pred_rasch= np.exp(-res["test_pass_datk_est3s"])
    
    mse_dist  = mean_squared_error(gt, pred_dist)
    mse_rasch = mean_squared_error(gt, pred_rasch)
    
    results_by_scenario.setdefault(scenario, []).append(
        (model_name, [mse_dist, mse_rasch])
    )
    
all_models = {
    model_name
    for models in results_by_scenario.values()
    for model_name, _ in models
}

ordered_scenarios = [
    "gsm",
    "mmlu",
    "lsat_qa",
    "legalbench",
    "bbq",
    "commonsense",
    "math",
    "med_qa",
    "legal_support"
]

# (3) Extract the full set of model names (from results_by_scenario) and sort them:
all_models = sorted({
    model_name
    for models_list in results_by_scenario.values()
    for model_name, _ in models_list
})

# (4) Build a DataFrame of shape (len(ordered_scenarios) × len(all_models)),
#     where each cell = (mse_dist − mse_rasch) or NaN if missing.
diff_matrix = []
for scenario in ordered_scenarios:
    # Create a mapping model_name → (mse_dist, mse_rasch) for this scenario
    scenario_dict = {m: vals for m, vals in results_by_scenario.get(scenario, [])}
    row_diffs = []
    for m in all_models:
        if m in scenario_dict:
            mse_dist, mse_rasch = scenario_dict[m]
            row_diffs.append(mse_dist - mse_rasch)
        else:
            row_diffs.append(np.nan)
    diff_matrix.append(row_diffs)

diff_df = pd.DataFrame(
    diff_matrix,
    index=ordered_scenarios,
    columns=all_models
)

sign_df = diff_df.apply(np.sign)

# (b) plot sign heatmap
plt.figure(figsize=(len(all_models) * 0.5 + 3, len(ordered_scenarios) * 0.5 + 3))
sns.set_style("white")

ax = sns.heatmap(
    sign_df,
    cmap="bwr",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="lightgray",
    cbar_kws={"label": "sign(Distributional MSE − IRT MSE)"},
    annot=True,
    fmt=".0f",
    annot_kws={"fontsize": 8}
)

ax.set_xlabel("Model")
ax.set_ylabel("Dataset")
ax.set_title("Sign of Test MSE Difference (Distributional − IRT)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("monkey_mse_sign_heatmap.png", dpi=300, bbox_inches="tight")

# # (5) Create a colormap that maps NaN → dark yellow:
# cmap = plt.get_cmap("bwr").copy()
# # Mark any NaN as “bad” and color it dark yellow
# cmap.set_bad(color="darkgoldenrod")

# # (6) Plot the heatmap:
# plt.figure(figsize=(len(all_models) * 0.5 + 3, len(ordered_scenarios) * 0.5 + 3))
# sns.set_style("white")

# # Determine the max absolute value, to center at zero
# abs_max = np.nanmax(np.abs(diff_df.values))

# ax = sns.heatmap(
#     diff_df,
#     cmap=cmap,
#     center=0,
#     vmin=-abs_max,
#     vmax=abs_max,
#     linewidths=0.5,
#     linecolor="lightgray",
#     cbar_kws={"label": "Distributional MSE − IRT MIRTzSE"},
#     annot=True,
#     fmt=".2f",
#     annot_kws={"fontsize": 8},
#     mask=False  # we rely on cmap.set_bad to color NaNs
# )

# ax.set_xlabel("Model")
# ax.set_ylabel("Dataset")
# ax.set_title("Difference in Test MSE (Distributional − IRT)")

# plt.xticks(rotation=45, ha="right")
# plt.yticks(rotation=0)
# plt.tight_layout()

# # (7) Optionally save to a file
# plt.savefig("monkey_mse_difference_heatmap.png", dpi=300, bbox_inches="tight")
# plt.show()