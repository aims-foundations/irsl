import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tueplots import bundles

# Load results
with open("downstream_data.pkl", "rb") as f:
    results_dict = pickle.load(f)

# Output directory
out_dir = "result/downstream"
os.makedirs(out_dir, exist_ok=True)

for scenario, model_results in results_dict.items():
    models = list(model_results.keys())

    # Compute MSEs
    linear_train_mse = []
    irt_train_mse    = []
    linear_test_mse  = []
    irt_test_mse     = []
    for m in models:
        res = model_results[m]
        linear_train_mse.append(mean_squared_error(res["gt_ctt_train"], res["train_linears"]))
        irt_train_mse.append(   mean_squared_error(res["gt_ctt_train"], res["train_irts"]))
        linear_test_mse.append( mean_squared_error(res["gt_ctt_test"],  res["train_linears"]))
        irt_test_mse.append(    mean_squared_error(res["gt_ctt_test"],  res["test_irts"]))

    # X-axis
    x = np.arange(len(models))
    width = 0.35

    # One plot per scenario
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(10, 4))

        # Overlapping bars for Train at x - width/2
        ax.bar(x - width/2, linear_train_mse, width=width, alpha=0.6, label="Linear (Train)")
        ax.bar(x - width/2, irt_train_mse,    width=width, alpha=0.6, label="IRT (Train)")

        # Overlapping bars for Test at x + width/2
        ax.bar(x + width/2, linear_test_mse, width=width, alpha=0.6, label="Linear (Test)")
        ax.bar(x + width/2, irt_test_mse,    width=width, alpha=0.6, label="IRT (Test)")

        # Labels & styling
        ax.set_ylabel("MSE", fontsize=16)
        ax.set_title(f"{scenario}", fontsize=18)
        ax.set_xticks(x)
        models = [m.split("-intermediate-checkpoints")[0] if m.endswith("-intermediate-checkpoints") else m for m in models]
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=12, loc="upper right")

        plt.tight_layout()
        fig.savefig(f"{out_dir}/downstream_mse_{scenario}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

for scenario, model_results in results_dict.items():
    for m, res in model_results.items():
        # extract the FLOPs and all four curves
        flops         = res["flops"]          # assumed shape (K,)
        gt_ctt_train  = res["gt_ctt_train"]   # shape (K,)
        gt_ctt_test   = res["gt_ctt_test"]    # shape (K,)
        train_linears = res["train_linears"]  # rename your linear fits
        train_irts    = res["train_irts"]
        test_irts     = res["test_irts"]

        # clean up model name for file
        clean_name = m.split("-intermediate-checkpoints")[0]

        # make output subdir
        save_dir = os.path.join(out_dir, scenario)
        os.makedirs(save_dir, exist_ok=True)

        # two‐panel figure
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # ————— Left: Train
            ax = axes[0]
            ax.plot(flops, gt_ctt_train,
                    linestyle='-',
                    color='black',
                    linewidth=2,
                    label='Ground truth')
            ax.plot(flops, train_linears,
                    linestyle='--',
                    label='Kaplan')
            ax.plot(flops, train_irts,
                    linestyle='--',
                    label='Rasch')
            ax.set_xlabel("FLOPs (1e21)", fontsize=20)
            ax.set_ylabel("Mean MMLU Accuracy", fontsize=20)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=14)
            ax.set_title('Train', fontsize=16)

            # ————— Right: Test
            ax = axes[1]
            ax.plot(flops, gt_ctt_test,
                    linestyle='-',
                    color='black',
                    linewidth=2,
                    label='Ground truth')
            ax.plot(flops, train_linears,  # reuse the linear fit
                    linestyle='--',
                    label='Kaplan')
            ax.plot(flops, test_irts,
                    linestyle='--',
                    label='Rasch')
            ax.set_xlabel("FLOPs (1e21)", fontsize=20)
            ax.set_ylabel("Mean MMLU Accuracy", fontsize=20)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=14)
            ax.set_title('Test', fontsize=16)

            fig.tight_layout()
            fig.savefig(f"{save_dir}/downstream_{clean_name}.png",
                        dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
