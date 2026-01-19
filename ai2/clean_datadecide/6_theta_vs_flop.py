from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tueplots import bundles
bundles.icml2024()

base_dir = Path(__file__).resolve().parent
input_path = base_dir / "data" / "6_long.parquet"
output_dir = base_dir / "results" / "6_theta_vs_flop"
MAX_STEP_PERCENTAGE = 0.1

df = pd.read_parquet(input_path)
ability_cols = [c for c in df.columns if c.startswith("ability_") and c.endswith("_binary")]
bench_names = sorted({c[len("ability_") : -len("_binary")] for c in ability_cols})
mixes = sorted(df["model_data_mix"].unique())

for bench in bench_names:
    binary_col = f"ability_{bench}_binary"
    beta_col = f"ability_{bench}_prob"
    bench_dir = output_dir / bench
    bench_dir.mkdir(parents=True, exist_ok=True)

    for mix in mixes:
        df_mix = df[df["model_data_mix"] == mix]
        for kind, col in (("binary", binary_col), ("beta", beta_col)):
            plot_df = df_mix.loc[df_mix["FLOP"].notna(), ["FLOP", col]]

            avg_rows = []
            for _, group in df_mix.groupby(["model_size"]):
                group = group.loc[:, ["model_step", "FLOP", col]]
                group = group.sort_values("model_step")
                top_n = int(np.ceil(len(group) * MAX_STEP_PERCENTAGE))
                avg_rows.append(
                    {
                        "FLOP": group["FLOP"].iloc[-1],
                        col: float(group[col].tail(top_n).mean()),
                    }
                )
            avg_df = pd.DataFrame(avg_rows)

            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), sharey=True)
                ax_linear, ax_log = axes[0]
                ax_avg_linear, ax_avg_log = axes[1]

                ax_linear.scatter(plot_df["FLOP"], plot_df[col], s=70, alpha=0.7)
                ax_linear.set_xlabel("FLOP", fontsize=16)
                ax_linear.set_ylabel(r"$\theta$", fontsize=16)
                ax_linear.set_title(r"Final $\theta$", fontsize=18)
                ax_linear.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
                ax_linear.tick_params(axis="both", labelsize=14)

                ax_log.scatter(plot_df["FLOP"], plot_df[col], s=70, alpha=0.7)
                ax_log.set_xlabel("FLOP (log scale)", fontsize=16)
                ax_log.set_xscale("log")
                ax_log.set_title(r"Final $\theta$, Log Scale", fontsize=18)
                ax_log.tick_params(axis="both", labelsize=14)

                ax_avg_linear.scatter(avg_df["FLOP"], avg_df[col], s=70, alpha=0.7)
                ax_avg_linear.set_xlabel("FLOP", fontsize=16)
                ax_avg_linear.set_ylabel(r"$\theta$", fontsize=16)
                ax_avg_linear.set_title(r"Avg Final 10\% $\theta$", fontsize=18)
                ax_avg_linear.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
                ax_avg_linear.tick_params(axis="both", labelsize=14)

                ax_avg_log.scatter(avg_df["FLOP"], avg_df[col], s=70, alpha=0.7)
                ax_avg_log.set_xlabel("FLOP (log scale)", fontsize=16)
                ax_avg_log.set_xscale("log")
                ax_avg_log.set_title(r"Avg Final 10\% $\theta$, Log Scale", fontsize=18)
                ax_avg_log.tick_params(axis="both", labelsize=14)

                fig.suptitle(f"{bench}, {mix}, {kind}", fontsize=18)
                fig.tight_layout()
                out_path = bench_dir / f"{kind}_{mix}_{bench}_theta_vs_flop.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
