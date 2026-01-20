from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tueplots import bundles
bundles.icml2024()
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))
from utils import recursive_defaultdict

LONG_INPUT_PATH = BASE_DIR / "data" / "6_long.parquet"
DICT_INPUT_PATH = BASE_DIR / "data" / "7_filt_laws.pkl"
DECISION_ACC_OUTPUT_DIR = BASE_DIR / "results" / "8_plot" / "decision_accuracy"

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
    DECISION_ACC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_input = pd.read_parquet(LONG_INPUT_PATH)
    target_flop = df_input["FLOP"].max()
    target_df = df_input[df_input["FLOP"] == target_flop]
    assert len(target_df) == 20
    with open(DICT_INPUT_PATH, "rb") as f:
        input_dict = pickle.load(f)

    for bench, bench_dict in tqdm(input_dict.items(), desc="bench"):
        gt_rank = rank_mix(target_df, f"acc_full_{bench}")

        bench_rows = []
        for mix, mix_dict in bench_dict.items():
            for max_size, size_dict in mix_dict.items():
                bench_rows.append(
                    {
                        "flop_ratio": size_dict["max_flop"] / target_flop,
                        "max_model_size": max_size,
                        "model_data_mix": mix,
                        "classic_acc": size_dict["classic"]["pred_1B"]["acc"],
                        "classic_prob": size_dict["classic"]["pred_1B"]["prob"],
                        "irt_acc": size_dict["irt"]["pred_1B"]["acc"],
                        "irt_prob": size_dict["irt"]["pred_1B"]["prob"],
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
            plot_rows.append(
                {
                    "flop_ratio": flop_ratio,
                    "classic_acc_decisionacc": classic_acc_decisionacc,
                    "classic_prob_decisionacc": classic_prob_decisionacc,
                    "irt_acc_decisionacc": irt_acc_decisionacc,
                    "irt_prob_decisionacc": irt_prob_decisionacc,
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
                
