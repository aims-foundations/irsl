import pandas as pd
from tqdm import tqdm
import torch
from torch.distributions import Bernoulli
from torch.optim import LBFGS
import pickle
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import numpy as np
np.random.seed(0)
torch.manual_seed(42)
from scipy.optimize import curve_fit
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.kernel_ridge import KernelRidge
from pathlib import Path
import glob
from huggingface_hub import snapshot_download
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    device = "cuda:7"
    
    # ScalingIntelligence/monkey_business
    monkey_business_list = [
        f"hf://datasets/ScalingIntelligence/monkey_business/GSM8K_{monkey_model}.json"
        for monkey_model in ["Llama-3-8B-Instruct", "Llama-3-70B-Instruct"]
    ]
    
    # Rylan query
    local_dir = snapshot_download(
        repo_id="stair-lab/monkey_queries",
        repo_type="dataset"
    )
    rylan_query_list = glob.glob(os.path.join(local_dir, "rylan_query", "GSM8K*"))
    
    # we query
    benchmark2scenario = {
        "lite": ["legalbench", "math", "commonsense", "med_qa", "gsm"],
        "mmlu": ["mmlu"],
        "classic": ["bbq", "lsat_qa"] # , "legal_support"
    }
    we_query_list = []
    for pattern in ("*mmlu.json", "*lsat_qa.json", "*legalbench.json"):
        we_query_list.extend(glob.glob(os.path.join(local_dir, pattern)))
    
    # harmbench
    harmbench_model_names = [
        "GraySwanAI-Llama-3-8B-Instruct-RR",
        "claude-3-5-sonnet-20240620",
        "cygnet",
        "gpt-4o-mini",
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "meta-llama-Meta-Llama-3-8B-Instruct",
        "gpt-4o",
        "claude-3-opus-20240229",
    ]
    harmbench_list = [
        f"hf://datasets/stair-lab/monkey_queries/{harmbench_model_name}_harm_bench.json"
        for harmbench_model_name in harmbench_model_names
    ]
    
    scenario_to_benchmark = {
        **{
            scenario: benchmark_name
            for benchmark_name, scenario_list in benchmark2scenario.items()
            for scenario in scenario_list
        },
        'harm_bench': 'safety',
        'gsm': 'lite',
    }
    
    all_list = monkey_business_list + we_query_list + rylan_query_list + harmbench_list
    for path in all_list:
        stem = Path(path).stem # e.g. "pythia-12b_lsat_qa"
        monkey_model_name, scenario_name = stem.split("_", 1)
        if monkey_model_name == "GSM8K":
            monkey_model_name = scenario_name
            scenario_name = "gsm"

        benchmark_name = scenario_to_benchmark.get(scenario_name)
        print(f"\nmodel={monkey_model_name}, scenario={scenario_name}, benchmark={benchmark_name}")
        guess = 0
        monkey_dataset = pd.read_json(path)
        # monkey_dataset = pd.read_json(f"hf://datasets/stair-lab/monkey_queries/{monkey_model_name}_{scenario_name}.json")
        
        pkl_name = "results_with_z" if benchmark_name != "safety" else "results_with_z_harmbench"
        with open(f"/lfs/skampere1/0/sttruong/deval/data/gather_helm_data/{pkl_name}.pkl", "rb") as f:
            helm_resmat_full = pickle.load(f)
        keep_cols = ~helm_resmat_full.columns.get_level_values("z").isna()
        helm_resmat_full = helm_resmat_full.loc[:, keep_cols]
        
        output_dir = f"result/monkey_passat1"
        os.makedirs(output_dir, exist_ok=True)
        
        monkey_questions2iscorrects = {row["question"]: row["is_corrects"] for _, row in monkey_dataset.iterrows()}
        print(f"len(monkey_questions2iscorrects): {len(monkey_questions2iscorrects)}")
        lengths = [len(l) for l in monkey_questions2iscorrects.values()]
        print(f"set(lengths): {set(lengths)}")
        
        helm_resmat = helm_resmat_full.loc[:, helm_resmat_full.columns.get_level_values("benchmark") == benchmark_name]
        helm_resmat = helm_resmat.loc[:, helm_resmat.columns.get_level_values("scenario") == scenario_name]
        helm_resmat = helm_resmat[~helm_resmat.isna().all(axis=1)]
        print(f"helm_resmat.shape: {helm_resmat.shape}")
        helm_questions = list(helm_resmat.columns.get_level_values("input.text"))

        helm_question_set = set(helm_questions)
        monkey_question_set = set(monkey_questions2iscorrects.keys())
        intersect_questions = sorted(helm_question_set & monkey_question_set)
        print(f"interset: {len(intersect_questions)}")
        filtered_columns = [
            col for q in intersect_questions
            for col in helm_resmat.columns
            if col[helm_resmat.columns.names.index("input.text")] == q
        ]
        helm_resmat = helm_resmat.loc[:, filtered_columns]
        monkey_questions2iscorrects = {q: monkey_questions2iscorrects[q] for q in intersect_questions}
        
        train_pass_iat1s = np.array([sum(iscorrects)/len(iscorrects) for iscorrects in monkey_questions2iscorrects.values()])
        
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            plt.figure()
            plt.hist(
                train_pass_iat1s,
                bins=50,            # adjust number of bins as needed
                density=True,       # normalize to form a density
                alpha=0.7,          # make bars slightly transparent
                edgecolor="purple"   # outline each bar
            )
            plt.xlabel("Pass at 1")  # x‐axis label
            plt.ylabel("Density")    # y‐axis label

            plt.savefig(
                f"{output_dir}/passat1_{monkey_model_name}_{scenario_name}.png",
                dpi=300
            )
            plt.close()