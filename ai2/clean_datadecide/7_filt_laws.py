import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import sys
from scipy.special import expit
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent / "data"
sys.path.append(str(BASE_DIR.parent.parent.parent))
from utils import (
    MODEL2PARA,
    recursive_defaultdict,
    fn_step1_classic,
    fit_step1_classic,
    fn_step2_classic,
    fit_step2_classic,
    fn_step1_irt,
    fit_step1_irt,
)

# function forms
# - classic:
#   - step1: 
#     - correct_bpb_sub VS FLOP: f1, fit_step1_classic
#   - step2:
#     - acc_per_char_sub VS correct_bpb_sub: f21, fit_step2_classic
#     - p_correct_choice_sub VS correct_bpb_sub: f22, fit_step2_classic
# - IRT:
#   - step1:
#     - theta_binary VS FLOP: g12, fit_step1_irt
#     - theta_beta VS FLOP: g12, fit_step1_irt

# output_dict = {
#     bench (str): {
#         model_data_mix (str): {
#             max_model_size (int): {
#                 "max_flop": float,
#                 "classic": {
#                     "data": {
#                         "step1": [(FLOP (float), bpb (float)), ...],
#                         "step2": [(bpb (float), acc (float), prob (float)), ...],
#                     },
#                     "paras": {
#                         "step1": f1_paras (list of float, len = 3),
#                         "step2_acc": f21_paras (list of float, len = 4),
#                         "step2_prob": f22_paras (list of float, len = 4),
#                     },
#                     "pred_1B": {
#                         "acc": float,
#                         "prob": float,
#                     }
#                 },
#                 "irt": {
#                     "data": {
#                         "step1": [(FLOP (float), theta_binary (float), theta_beta (float)), ...],
#                     },
#                     "paras": {
#                         "step1_binary": g11_paras (list of float, len = 2),
#                         "step1_beta": g12_paras (list of float, len = 2),
#                     },
#                     "pred_1B": {
#                         "acc": float,
#                         "prob": float,
#                     }
#                 }
#             }
#         }
#     }
# }

INPUT_PATH = BASE_DIR / "6_long.parquet"
DIFF_INPUT_PATH = BASE_DIR / "6_difficulty.parquet"
OUTPUT_PATH = BASE_DIR / "7_filt_laws.pkl"

if __name__ == "__main__":
    output_dict = recursive_defaultdict()
    
    df_difficulty = pd.read_parquet(DIFF_INPUT_PATH)
    binary_difficulty = df_difficulty["binary_difficulty"].to_numpy()
    beta_difficulty = df_difficulty["beta_difficulty"].to_numpy()
    
    df_input = pd.read_parquet(INPUT_PATH)
    df_input["model_size"] = df_input["model_size"].map(MODEL2PARA).astype(int)
    unique_mixes = sorted(df_input["model_data_mix"].unique())
    unique_model_sizes = sorted(int(v) for v in df_input["model_size"].unique())
    binary_theta_cols = [c for c in df_input.columns if c.startswith("ability_") and c.endswith("_binary")]
    unique_benches = sorted({c[len("ability_") : -len("_binary")] for c in binary_theta_cols})
    max_flop = df_input["FLOP"].max() # 1B, final step
    
    for mix in tqdm(unique_mixes, desc="mix"):
        df_mix = df_input[df_input["model_data_mix"] == mix]
        for max_size in tqdm(unique_model_sizes[1:-1], desc="max_size", leave=False): # removing first and last
            df_size = df_mix[df_mix["model_size"] <= max_size]
            max_size_flop = df_size["FLOP"].max()
            for bench in tqdm(unique_benches, desc="bench", leave=False):
                output_dict[bench][mix][max_size]["max_flop"] = float(max_size_flop)
                
                # fit classic step 1
                data_classic_step1 = df_size.loc[
                    df_size["FLOP"].notna() & df_size[f"correct_bpb_sub_{bench}"].notna(),
                    ["FLOP", f"correct_bpb_sub_{bench}"],
                ]
                output_dict[bench][mix][max_size]["classic"]["data"]["step1"] = list(
                    data_classic_step1.itertuples(index=False, name=None)
                )
                
                f1_paras = fit_step1_classic(
                    flops=data_classic_step1["FLOP"].tolist(),
                    bpbs=data_classic_step1[f"correct_bpb_sub_{bench}"].tolist(),
                )
                output_dict[bench][mix][max_size]["classic"]["paras"]["step1"] = f1_paras
                
                # fit irt step 1
                data_irt_step1 = df_size.loc[
                    df_size["FLOP"].notna(),
                    ["FLOP", f"ability_{bench}_binary", f"ability_{bench}_prob"],
                ]
                output_dict[bench][mix][max_size]["irt"]["data"]["step1"] = list(
                    data_irt_step1.itertuples(index=False, name=None)
                )
                
                g11_paras = fit_step1_irt(
                    flops=data_irt_step1["FLOP"].tolist(),
                    thetas=data_irt_step1[f"ability_{bench}_binary"].tolist(),
                )
                output_dict[bench][mix][max_size]["irt"]["paras"]["step1_binary"] = g11_paras
                g12_paras = fit_step1_irt(
                    flops=data_irt_step1["FLOP"].tolist(),
                    thetas=data_irt_step1[f"ability_{bench}_prob"].tolist(),
                )
                output_dict[bench][mix][max_size]["irt"]["paras"]["step1_beta"] = g12_paras
                
                # fit classic step 2
                data_classic_step2 = df_size.loc[
                    df_size[f"correct_bpb_sub_{bench}"].notna(),
                    [f"correct_bpb_sub_{bench}", f"acc_sub_{bench}", f"p_correct_choice_sub_{bench}"],
                ]
                output_dict[bench][mix][max_size]["classic"]["data"]["step2"] = list(
                    data_classic_step2.itertuples(index=False, name=None)
                )
                
                f21_paras = fit_step2_classic(
                    bpbs=data_classic_step2[f"correct_bpb_sub_{bench}"].tolist(),
                    metrics=data_classic_step2[f"acc_sub_{bench}"].tolist(),
                )
                output_dict[bench][mix][max_size]["classic"]["paras"]["step2_acc"] = f21_paras
                f22_paras = fit_step2_classic(
                    bpbs=data_classic_step2[f"correct_bpb_sub_{bench}"].tolist(),
                    metrics=data_classic_step2[f"p_correct_choice_sub_{bench}"].tolist(),
                )
                output_dict[bench][mix][max_size]["classic"]["paras"]["step2_prob"] = f22_paras
                
                # extrapolate
                pred_acc_classic = fn_step2_classic(
                    bpb=fn_step1_classic(flop=max_flop, paras=f1_paras),
                    paras=f21_paras,
                )
                pred_prob_classic = fn_step2_classic(
                    bpb=fn_step1_classic(flop=max_flop, paras=f1_paras),
                    paras=f22_paras,
                )
                pred_theta_binary = fn_step1_irt(flop=max_flop, paras=g11_paras)
                pred_theta_beta = fn_step1_irt(flop=max_flop, paras=g12_paras)
                pred_acc_irt = expit(pred_theta_binary + binary_difficulty).mean()
                pred_prob_irt = expit(pred_theta_beta + beta_difficulty).mean()

                output_dict[bench][mix][max_size]["classic"]["pred_1B"]["acc"] = float(pred_acc_classic)
                output_dict[bench][mix][max_size]["classic"]["pred_1B"]["prob"] = float(pred_prob_classic)
                output_dict[bench][mix][max_size]["irt"]["pred_1B"]["acc"] = float(pred_acc_irt)
                output_dict[bench][mix][max_size]["irt"]["pred_1B"]["prob"] = float(pred_prob_irt)
    
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output_dict, f)
