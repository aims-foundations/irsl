import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tueplots import bundles

# Load results
with open("mmlu_mmlu_curve_results.pkl", "rb") as f:
    results_dict = pickle.load(f)

benchmark = "mmlu"
scenario = "mmlu"
models = list(results_dict[benchmark][scenario].keys())

# Compute MSEs
linear_train_mse, irt_train_mse = [], []
linear_test_mse, irt_test_mse = [], []

for model in models:
    res = results_dict[benchmark][scenario][model]
    linear_train_mse.append(mean_squared_error(res["gt_ctt_train"], res["train_linears"]))
    irt_train_mse.append(mean_squared_error(res["gt_ctt_train"], res["train_irts"]))
    linear_test_mse.append(mean_squared_error(res["gt_ctt_test"], res["train_linears"]))
    irt_test_mse.append(mean_squared_error(res["gt_ctt_test"], res["test_irts"]))

# Prepare data
x = np.arange(len(models))  # model index
width = 0.35  # width of train vs. test group

with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot bars and store the returned BarContainers
    bars = []
    bars.append(ax.bar(x - width/2, linear_train_mse, width=width, alpha=0.6, label="Linear (Train)"))
    bars.append(ax.bar(x - width/2, irt_train_mse, width=width, alpha=0.6, label="IRT (Train)"))

    bars.append(ax.bar(x + width/2, linear_test_mse, width=width, alpha=0.6, label="Linear (Test)"))
    bars.append(ax.bar(x + width/2, irt_test_mse, width=width, alpha=0.6, label="IRT (Test)"))

    # # Add value labels
    # for bar_group in bars:
    #     for bar in bar_group:
    #         height = bar.get_height()
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             height,
    #             f"{height:.2e}",
    #             ha="center",
    #             va="bottom",
    #             fontsize=8,
    #             rotation=0  # horizontal
    #         )

    # Labels & styling
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_title("Train/Test MSE: Linear vs. IRT", fontsize=16)
    ax.set_xticks(x)
    breakpoint()
    models = ["SmolLM2-1.7B" if m=="SmolLM2-1.7B-intermediate-checkpoints" else m for m in models]
    ax.set_xticklabels(models, rotation=20)
    ax.legend(fontsize=6)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("downstream_mse.png", dpi=300)
    plt.show()