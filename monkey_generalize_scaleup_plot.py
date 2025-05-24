import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tueplots import bundles
import glob

# monkey_model_names = ["Meta-Llama-3-8B-Instruct", "pythia-12b", "pythia-6.9b"]
# for monkey_model_name in monkey_model_names:
#     scenarios_all = ["commonsense", "math", "med_qa", "legalbench", "mmlu", "bbq"] # "legal_support", "lsat_qa" 

#     test_mse = []
#     for scenario in scenarios_all:
    #     p = f"result/monkey_generalize_scaleup_diff_split/monkey_scaleup_data_{monkey_model_name}_{scenario}.pkl"
    #     with open(p, "rb") as f:
    #         res = pickle.load(f)

    #     # invert the negative log values back to probabilities
    #     gt        = np.exp(-res["test_neglog_gts"])
    #     pred_ls   = np.exp(-res["train_neglog_est_1"])
    #     pred_dist = np.exp(-res["train_neglog_est_2"])
    #     pred_rasch= np.exp(-res["test_neglog_est_3"])

    #     mse_ls    = mean_squared_error(gt, pred_ls)
    #     mse_dist  = mean_squared_error(gt, pred_dist)
    #     mse_rasch = mean_squared_error(gt, pred_rasch)

    #     test_mse.append([mse_ls, mse_dist, mse_rasch])

    # arr = np.array(test_mse)
    # x = np.arange(len(scenarios_all))
    # width = 0.2

    # with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     ax.bar(x - width,     arr[:, 0], width=width, label="Least squares",    alpha=0.8)
    #     ax.bar(x,             arr[:, 1], width=width, label="Distributional",   alpha=0.8)
    #     ax.bar(x + width,     arr[:, 2], width=width, label="Rasch",            alpha=0.8)

    #     ax.set_ylabel("Test MSE", fontsize=16)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(scenarios_all, rotation=45, ha="right", fontsize=12)
    #     ax.tick_params(axis="both", labelsize=12)
    #     ax.legend(fontsize=12, loc="upper right")
    #     ax.set_yscale("log")
        
    #     plt.tight_layout()
    #     fig.savefig(
    #         os.path.join(f"monkey_scaleup_{monkey_model_name}.png"),
    #         dpi=300,
    #         bbox_inches="tight"
    #     )
    #     plt.close(fig)

# 1) Define your fixed scenario
scenario = "gsm"

# 2) Find all matching .pkl files for that scenario
pattern = f"result/monkey_generalize_scaleup_diff_split/monkey_scaleup_data_*_{scenario}.pkl"
filepaths = glob.glob(pattern)

test_mse = []
model_names = []

for fp in filepaths:
    # 3) Extract monkey_model_name from filename
    fname = os.path.basename(fp)  # e.g. monkey_scaleup_data_Llama-3-8B_gsm.pkl
    monkey_model_name = fname.split("monkey_scaleup_data_")[1].split(".pkl")[0]
    model_names.append(monkey_model_name)

    # 4) Load results and invert neglog back to probabilities
    with open(fp, "rb") as f:
        res = pickle.load(f)

    gt        = np.exp(-res["test_neglog_gts"])
    pred_ls   = np.exp(-res["train_neglog_est_1"])
    pred_dist = np.exp(-res["train_neglog_est_2"])
    pred_rasch= np.exp(-res["test_neglog_est_3"])

    # 5) Compute MSEs
    mse_ls    = mean_squared_error(gt, pred_ls)
    mse_dist  = mean_squared_error(gt, pred_dist)
    mse_rasch = mean_squared_error(gt, pred_rasch)
    test_mse.append([mse_ls, mse_dist, mse_rasch])

# 6) Stack into array and order by strength of model if you like
arr = np.array(test_mse)
x = np.arange(len(model_names))
width = 0.25

# 7) Plot
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

    # 8) Save
    out_fname = f"monkey_scaleup_models_{scenario}.png"
    fig.savefig(out_fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
