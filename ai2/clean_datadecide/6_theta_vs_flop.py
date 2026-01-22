from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tueplots import bundles
bundles.icml2024()
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "6_long.parquet"
OUTPUT_DIR = BASE_DIR / "results" / "6_theta_vs_flop"
MAX_STEP_PERCENTAGE = 0.1

df = pd.read_parquet(INPUT_PATH)
ability_cols = [c for c in df.columns if c.startswith("ability_") and c.endswith("_binary")]
unique_benches = sorted({c[len("ability_") : -len("_binary")] for c in ability_cols})

for bench in unique_benches:
    binary_col = f"ability_{bench}_binary"
    beta_col = f"ability_{bench}_prob"
    bench_dir = OUTPUT_DIR / bench
    bench_dir.mkdir(parents=True, exist_ok=True)

    plot_df_binary = df.loc[df["FLOP"].notna(), ["FLOP", binary_col]]
    plot_df_beta = df.loc[df["FLOP"].notna(), ["FLOP", beta_col]]

    avg_rows_binary = []
    avg_rows_beta = []
    for _, group in df.groupby(["model_data_mix", "model_size"]):
        group = group.loc[:, ["model_step", "FLOP", binary_col, beta_col]]
        group = group.sort_values("model_step")
        top_n = int(np.ceil(len(group) * MAX_STEP_PERCENTAGE))
        avg_rows_binary.append(
            {
                "FLOP": group["FLOP"].iloc[-1],
                binary_col: float(group[binary_col].tail(top_n).mean()),
            }
        )
        avg_rows_beta.append(
            {
                "FLOP": group["FLOP"].iloc[-1],
                beta_col: float(group[beta_col].tail(top_n).mean()),
            }
        )
    avg_df_binary = pd.DataFrame(avg_rows_binary)
    avg_df_beta = pd.DataFrame(avg_rows_beta)

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), sharey=True)
        ax_linear, ax_log = axes[0]
        ax_avg_linear, ax_avg_log = axes[1]

        ax_linear.scatter(plot_df_binary["FLOP"], plot_df_binary[binary_col], s=80, alpha=0.7, label="binary")
        ax_linear.scatter(plot_df_beta["FLOP"], plot_df_beta[beta_col], s=80, alpha=0.7, label="beta")
        ax_linear.set_xlabel("FLOP", fontsize=20)
        ax_linear.set_ylabel(r"$\theta$", fontsize=20)
        ax_linear.set_title(r"Final $\theta$", fontsize=22)
        ax_linear.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax_linear.tick_params(axis="both", labelsize=18)
        ax_linear.legend(fontsize=16)

        ax_log.scatter(plot_df_binary["FLOP"], plot_df_binary[binary_col], s=80, alpha=0.7)
        ax_log.scatter(plot_df_beta["FLOP"], plot_df_beta[beta_col], s=80, alpha=0.7)
        ax_log.set_xlabel("FLOP (log scale)", fontsize=20)
        ax_log.set_xscale("log")
        ax_log.set_title(r"Final $\theta$, Log Scale", fontsize=22)
        ax_log.tick_params(axis="both", labelsize=18)

        ax_avg_linear.scatter(avg_df_binary["FLOP"], avg_df_binary[binary_col], s=80, alpha=0.7, label="binary")
        ax_avg_linear.scatter(avg_df_beta["FLOP"], avg_df_beta[beta_col], s=80, alpha=0.7, label="beta")
        ax_avg_linear.set_xlabel("FLOP", fontsize=20)
        ax_avg_linear.set_ylabel(r"$\theta$", fontsize=20)
        ax_avg_linear.set_title(r"Avg Final 10\% $\theta$", fontsize=22)
        ax_avg_linear.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax_avg_linear.tick_params(axis="both", labelsize=18)
        ax_avg_linear.legend(fontsize=16)

        ax_avg_log.scatter(avg_df_binary["FLOP"], avg_df_binary[binary_col], s=80, alpha=0.7)
        ax_avg_log.scatter(avg_df_beta["FLOP"], avg_df_beta[beta_col], s=80, alpha=0.7)
        ax_avg_log.set_xlabel("FLOP (log scale)", fontsize=20)
        ax_avg_log.set_xscale("log")
        ax_avg_log.set_title(r"Avg Final 10\% $\theta$, Log Scale", fontsize=22)
        ax_avg_log.tick_params(axis="both", labelsize=18)

        fig.suptitle(f"{bench}", fontsize=22)
        fig.tight_layout()
        out_path = bench_dir / f"{bench}_theta_vs_flop.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
