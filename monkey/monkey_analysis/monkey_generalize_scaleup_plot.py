import os
import pickle
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tueplots import bundles

# 1) find all files
dir = "result/monkey_generalize_scaleup_diff_split_mseloss"
pattern = f"{dir}/monkey_scaleup_data_*.pkl"
filepaths = glob.glob(pattern)

# 2) group results by scenario
results_by_scenario = {}
for fp in filepaths:
    fname = os.path.basename(fp)
    # get the core: "monkey_scaleup_data_{model}_{scenario}.pkl" → "{model}_{scenario}"
    core = fname.removeprefix("monkey_scaleup_data_").removesuffix(".pkl")
    # split only on the last underscore to separate model vs. scenario
    parts = re.split(r'(?<=[BbM])_', core, maxsplit=1)
    if len(parts) == 2:
        model_name, scenario = parts
    else:
        # fallback: split at the first underscore, whichever it is
        model_name, scenario = core.split("_", 1)

    
    # load and compute MSEs
    with open(fp, "rb") as f:
        res = pickle.load(f)
    gt        = np.exp(-res["test_neglog_gts"])
    pred_ls   = np.exp(-res["train_neglog_est_1"])
    pred_dist = np.exp(-res["train_neglog_est_2"])
    pred_rasch= np.exp(-res["test_neglog_est_3"])
    
    mse_ls    = mean_squared_error(gt, pred_ls)
    mse_dist  = mean_squared_error(gt, pred_dist)
    mse_rasch = mean_squared_error(gt, pred_rasch)
    
    # store
    results_by_scenario.setdefault(scenario, []).append(
        (model_name, [mse_ls, mse_dist, mse_rasch])
    )

# 3) loop over scenarios and plot
for scenario, model_data in results_by_scenario.items():
    # sort by model name (optional)
    model_data.sort(key=lambda x: x[0])
    model_names, mse_vals = zip(*model_data)
    arr = np.array(mse_vals)
    
    x = np.arange(len(model_names))
    width = 0.25
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width,     arr[:, 0], width=width, label="Least squares",  alpha=0.8)
        ax.bar(x,             arr[:, 1], width=width, label="Distributional", alpha=0.8)
        ax.bar(x + width,     arr[:, 2], width=width, label="Rasch",          alpha=0.8)

        ax.set_ylabel("Test MSE", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=12, loc="upper right")
        ax.set_yscale("log")
        plt.tight_layout()

        out_fname = f"{dir}/aggregate_fig/monkey_scaleup_models_{scenario}.png"
        out_dir = os.path.dirname(out_fname)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
