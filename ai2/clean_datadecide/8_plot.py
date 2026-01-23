from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.special import expit
from scipy.stats import spearmanr
from tueplots import bundles
bundles.icml2024()
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))
from utils import (
    recursive_defaultdict,
    fn_step1_classic,
    fn_step2_classic,
    fn_step1_irt,
)

LONG_INPUT_PATH = BASE_DIR / "data" / "6_long.parquet"
DIFF_INPUT_PATH = BASE_DIR / "data" / "6_difficulty.pkl"
DICT_INPUT_PATH = BASE_DIR / "data" / "7_filt_laws.pkl"
PROB_MATRIX_PATH = BASE_DIR / "data" / "5_prob_matrix_cated.parquet"
DECISION_ACC_OUTPUT_DIR = BASE_DIR / "results" / "8_plot" / "decision_accuracy"
CURVE_FIT_OUTPUT_DIR = BASE_DIR / "results" / "8_plot" / "curve_fit"
DIFFICULTY_PLOT_OUTPUT_DIR = BASE_DIR / "results" / "8_plot" / "prob_corr"

def calculate_decisionacc(gt_rank, pred_rank):
    gt_pos = {name: i for i, name in enumerate(gt_rank)}
    pred_pos = {name: i for i, name in enumerate(pred_rank)}
    assert set(gt_pos) == set(pred_pos)
    total = 0
    match = 0
    for i in range(len(gt_rank)):
        for j in range(i + 1, len(gt_rank)):
            total += 1
            a, b = gt_rank[i], gt_rank[j]
            if (gt_pos[a] - gt_pos[b]) * (pred_pos[a] - pred_pos[b]) > 0:
                match += 1
    return match / total

def rank_mix(df, col_name):
    return df.set_index("model_data_mix")[col_name].sort_values(ascending=False).index.tolist()

if __name__ == "__main__":
    # 1. decision accuracy
    DECISION_ACC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_input = pd.read_parquet(LONG_INPUT_PATH)
    target_flop = df_input["FLOP"].max()
    target_df = df_input[df_input["FLOP"] == target_flop]
    assert len(target_df) == 20
    with open(DICT_INPUT_PATH, "rb") as f:
        input_dict = pickle.load(f)

    for bench, bench_dict in tqdm(input_dict.items(), desc="decision accuracy"):
        gt_rank = rank_mix(target_df, f"acc_full_{bench}")

        bench_rows = []
        for mix, mix_dict in bench_dict.items():
            for max_size, size_dict in mix_dict.items():
                bench_rows.append(
                    {
                        "flop_ratio": size_dict["max_size_flop"] / target_flop,
                        "max_model_size": max_size,
                        "model_data_mix": mix,
                        "classic_acc": size_dict["classic"]["pred_1B"]["acc"],
                        "classic_prob": size_dict["classic"]["pred_1B"]["prob"],
                        "irt_acc": size_dict["irt"]["pred_1B"]["acc"],
                        "irt_prob": size_dict["irt"]["pred_1B"]["prob"],
                        "irt_theta_binary": size_dict["irt"]["pred_1B"]["theta_binary"],
                        "irt_theta_beta": size_dict["irt"]["pred_1B"]["theta_beta"],
                    }
                )
        bench_df = pd.DataFrame(bench_rows)
        
        plot_rows = []
        for max_size in tqdm(bench_df["max_model_size"].unique(), desc="max_model_size", leave=False):
            flop_df = bench_df[bench_df["max_model_size"] == max_size]
            flop_ratio = flop_df["flop_ratio"].mean()
            classic_acc_rank = rank_mix(flop_df, "classic_acc")
            classic_acc_decisionacc = calculate_decisionacc(gt_rank, classic_acc_rank)
            classic_prob_rank = rank_mix(flop_df, "classic_prob")
            classic_prob_decisionacc = calculate_decisionacc(gt_rank, classic_prob_rank)
            irt_acc_rank = rank_mix(flop_df, "irt_acc")
            irt_acc_decisionacc = calculate_decisionacc(gt_rank, irt_acc_rank)
            irt_prob_rank = rank_mix(flop_df, "irt_prob")
            irt_prob_decisionacc = calculate_decisionacc(gt_rank, irt_prob_rank)
            irt_theta_binary_rank = rank_mix(flop_df, "irt_theta_binary")
            irt_theta_binary_decisionacc = calculate_decisionacc(gt_rank, irt_theta_binary_rank)
            irt_theta_beta_rank = rank_mix(flop_df, "irt_theta_beta")
            irt_theta_beta_decisionacc = calculate_decisionacc(gt_rank, irt_theta_beta_rank)
            plot_rows.append(
                {
                    "flop_ratio": flop_ratio,
                    "classic_acc_decisionacc": classic_acc_decisionacc,
                    "classic_prob_decisionacc": classic_prob_decisionacc,
                    "irt_acc_decisionacc": irt_acc_decisionacc,
                    "irt_prob_decisionacc": irt_prob_decisionacc,
                    "irt_theta_binary_decisionacc": irt_theta_binary_decisionacc,
                    "irt_theta_beta_decisionacc": irt_theta_beta_decisionacc,
                }
            )
        plot_df = pd.DataFrame(plot_rows)

        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["classic_acc_decisionacc"],
                marker="o",
                label="Classic Acc",
            )
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["classic_prob_decisionacc"],
                marker="o",
                label="Classic Prob",
            )
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["irt_acc_decisionacc"],
                marker="o",
                label="IRT Acc",
            )
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["irt_prob_decisionacc"],
                marker="o",
                label="IRT Prob",
            )
            # ax.plot(
            #     plot_df["flop_ratio"],
            #     plot_df["irt_theta_binary_decisionacc"],
            #     marker="o",
            #     label="Binary-IRT Theta ",
            # )
            # ax.plot(
            #     plot_df["flop_ratio"],
            #     plot_df["irt_theta_beta_decisionacc"],
            #     marker="o",
            #     label="Beta-IRT Theta ",
            # )
            ax.set_xlabel(r"Max $C$ for Predicting / Target $C$", fontsize=16)
            ax.set_ylabel("Decision accuracy", fontsize=16)
            ax.set_title(f"{bench}", fontsize=18)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=12)
            fig.tight_layout()
            fig.savefig(
                DECISION_ACC_OUTPUT_DIR / f"decision_accuracy_{bench}.png",
                dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
                
    # # 2. curve fit
    # for bench, bench_dict in tqdm(input_dict.items(), desc="curve fit"):
    #     for mix, mix_dict in tqdm(bench_dict.items(), desc="mix", leave=False):
    #         for max_size, size_dict in tqdm(mix_dict.items(), desc="max_size", leave=False):
    #             output_dir = CURVE_FIT_OUTPUT_DIR / bench / mix / f"max_size_{max_size}"
    #             output_dir.mkdir(parents=True, exist_ok=True)

    #             # classic step1
    #             classic_step1_data = size_dict["classic"]["data"]["step1"]
    #             flops = np.array([row[0] for row in classic_step1_data], dtype=float)
    #             bpbs = np.array([row[1] for row in classic_step1_data], dtype=float)
    #             f1_paras = size_dict["classic"]["paras"]["step1"]
    #             x_curve = np.logspace(np.log10(flops.min()), np.log10(flops.max()), 200)
    #             y_curve = fn_step1_classic(x_curve, f1_paras)

    #             with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    #                 fig, ax = plt.subplots(figsize=(8, 5))
    #                 ax.scatter(flops, bpbs, color="tab:blue", alpha=0.7)
    #                 ax.plot(x_curve, y_curve, color="tab:blue")
    #                 ax.set_xlabel("FLOP", fontsize=16)
    #                 ax.set_ylabel("Correct BPB", fontsize=16)
    #                 ax.set_title(f"Classic Step1: {bench}, {mix}, {max_size}", fontsize=18)
    #                 ax.set_xscale("log")
    #                 ax.tick_params(axis="both", labelsize=14)
    #                 fig.tight_layout()
    #                 fig.savefig(output_dir / "classic_step1.png", dpi=300, bbox_inches="tight")
    #                 plt.close(fig)

    #             # classic step2
    #             classic_step2_data = size_dict["classic"]["data"]["step2"]
    #             bpbs = np.array([row[0] for row in classic_step2_data], dtype=float)
    #             accs = np.array([row[1] for row in classic_step2_data], dtype=float)
    #             probs = np.array([row[2] for row in classic_step2_data], dtype=float)
    #             f21_paras = size_dict["classic"]["paras"]["step2_acc"]
    #             f22_paras = size_dict["classic"]["paras"]["step2_prob"]
    #             x_curve = np.linspace(bpbs.min(), bpbs.max(), 200)
    #             y_acc_curve = fn_step2_classic(x_curve, f21_paras)
    #             y_prob_curve = fn_step2_classic(x_curve, f22_paras)

    #             with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    #                 fig, ax = plt.subplots(figsize=(8, 5))
    #                 ax.scatter(bpbs, accs, color="tab:blue", alpha=0.7, label="Acc dps")
    #                 ax.scatter(bpbs, probs, color="tab:red", alpha=0.7, label="Prob dps")
    #                 ax.plot(x_curve, y_acc_curve, color="tab:blue", label="Acc Curve")
    #                 ax.plot(x_curve, y_prob_curve, color="tab:red", label="Prob Curve")
    #                 ax.set_xlabel("Correct BPB", fontsize=16)
    #                 ax.set_ylabel("Acc / Prob", fontsize=16)
    #                 ax.set_title(f"Classic Step2: {bench}, {mix}, {max_size}", fontsize=18)
    #                 ax.tick_params(axis="both", labelsize=14)
    #                 ax.legend(fontsize=12)
    #                 fig.tight_layout()
    #                 fig.savefig(output_dir / "classic_step2.png", dpi=300, bbox_inches="tight")
    #                 plt.close(fig)

    #             # irt step1
    #             irt_step1 = size_dict["irt"]["data"]["step1"]
    #             flops = np.array([row[0] for row in irt_step1], dtype=float)
    #             theta_binary = np.array([row[1] for row in irt_step1], dtype=float)
    #             theta_beta = np.array([row[2] for row in irt_step1], dtype=float)
    #             g11_paras = size_dict["irt"]["paras"]["step1_binary"]
    #             g12_paras = size_dict["irt"]["paras"]["step1_beta"]
    #             x_curve = np.logspace(np.log10(flops.min()), np.log10(flops.max()), 200)
    #             y_binary_curve = fn_step1_irt(x_curve, g11_paras)
    #             y_beta_curve = fn_step1_irt(x_curve, g12_paras)

    #             with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    #                 fig, ax = plt.subplots(figsize=(8, 5))
    #                 ax.scatter(flops, theta_binary, color="tab:blue", alpha=0.7, label="Binary dps")
    #                 ax.scatter(flops, theta_beta, color="tab:red", alpha=0.7, label="Beta dps")
    #                 ax.plot(x_curve, y_binary_curve, color="tab:blue", label="Binary Curve")
    #                 ax.plot(x_curve, y_beta_curve, color="tab:red", label="Beta Curve")
    #                 ax.set_xlabel("FLOP", fontsize=16)
    #                 ax.set_ylabel(r"$\theta$", fontsize=16)
    #                 ax.set_title(f"IRT Step1: {bench}, {mix}, {max_size}", fontsize=18)
    #                 ax.set_xscale("log")
    #                 ax.tick_params(axis="both", labelsize=14)
    #                 ax.legend(fontsize=12)
    #                 fig.tight_layout()
    #                 fig.savefig(output_dir / "irt_step1.png", dpi=300, bbox_inches="tight")
    #                 plt.close(fig)

    # # 3. prob correlation plot
    # with open(DIFF_INPUT_PATH, "rb") as f:
    #     difficulty_dict = pickle.load(f)

    # prob_df = pd.read_parquet(PROB_MATRIX_PATH)
    # prob_target_df = prob_df[
    #     (prob_df.index.get_level_values("model_split") == "test")
    #     & (prob_df.index.get_level_values("model_size") == "1B")
    #     & (
    #         prob_df.index.get_level_values("model_step").astype(int)
    #         == prob_df.index.get_level_values("model_step").astype(int).max()
    #     )
    # ].copy()
    # assert len(prob_target_df) == 20
    # bench_names = prob_target_df.columns.get_level_values("bench_name").map(
    #     lambda b: "mmlu" if b.startswith("mmlu") else b
    # )
    # zs = prob_target_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)

    # for bench, bench_dict in tqdm(input_dict.items(), desc="difficulty plot"):
    #     bench_mask = bench_names == bench
    #     bench_z = zs[bench_mask]

    #     for mix, mix_dict in tqdm(bench_dict.items(), desc="mix", leave=False):
    #         mix_prob = prob_target_df.xs(mix, level="model_data_mix").iloc[0].to_numpy(dtype=float)
    #         prob_gt = mix_prob[bench_mask]

    #         for max_size, size_dict in tqdm(mix_dict.items(), desc="max_size", leave=False):
    #             output_dir = DIFFICULTY_PLOT_OUTPUT_DIR / bench / mix / f"max_size_{max_size}"
    #             output_dir.mkdir(parents=True, exist_ok=True)

    #             theta_pred = fn_step1_irt(
    #                 flop=target_flop,
    #                 paras=size_dict["irt"]["paras"]["step1_beta"],
    #             )
    #             prob_pred = expit(theta_pred + bench_z)
    #             rho_beta, _ = spearmanr(prob_pred, prob_gt)

    #             with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    #                 fig = plt.figure(figsize=(6, 6))
    #                 gs = gridspec.GridSpec(5, 5, figure=fig, wspace=0.05, hspace=0.05)
    #                 ax_scatter = fig.add_subplot(gs[0:4, 1:5])
    #                 ax_left = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)
    #                 ax_bottom = fig.add_subplot(gs[4, 1:5], sharex=ax_scatter)

    #                 ax_scatter.scatter(prob_pred, prob_gt, s=10)
    #                 ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
    #                 ax_scatter.set_xlim(0, 1)
    #                 ax_scatter.set_ylim(0, 1)
    #                 ax_scatter.set_xlabel("IRT Probability", fontsize=16)
    #                 ax_scatter.set_ylabel("Empirical Probability", fontsize=16)
    #                 ax_scatter.tick_params(axis="both", labelsize=12)

    #                 ax_left.hist(prob_gt, bins=30, orientation="horizontal")
    #                 ax_left.set_ylim(0, 1)
    #                 ax_left.invert_xaxis()
    #                 ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    #                 ax_left.set_xticks([])
    #                 for spine in ("top", "right", "bottom", "left"):
    #                     ax_left.spines[spine].set_visible(False)

    #                 ax_bottom.hist(prob_pred, bins=30)
    #                 ax_bottom.set_xlim(0, 1)
    #                 ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    #                 ax_bottom.set_yticks([])
    #                 for spine in ("top", "right", "bottom", "left"):
    #                     ax_bottom.spines[spine].set_visible(False)

    #                 fig.suptitle(
    #                     rf"{bench}, {mix}, {max_size} ($\rho$ = {rho_beta:.2f})",
    #                     fontsize=14,
    #                 )
    #                 plt.subplots_adjust(wspace=0.05, hspace=0.05)
    #                 fig.savefig(output_dir / f"scatter_corr_beta_{mix}_{max_size}_{bench}.png", dpi=300, bbox_inches="tight")
    #                 plt.close(fig)
