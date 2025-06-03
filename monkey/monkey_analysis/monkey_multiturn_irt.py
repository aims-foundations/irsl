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

def trainer(parameters, optim, closure, n_iter=100, verbose=True):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for iteration in pbar:
        if iteration > 0:
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss})
            
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                break
    return parameters

if __name__ == "__main__":
    device = "cuda:7"
    B = 50000
    method = "diff_split"
    
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
        'harm_bench': 'safety',
    }
    
    output_dir = f"result/monkey_multiturn_irt_{method}"
    os.makedirs(output_dir, exist_ok=True)

    counts = []
    for i, path in enumerate(harmbench_list):
        stem = Path(path).stem
        monkey_model_name, scenario_name = stem.split("_", 1)

        benchmark_name = scenario_to_benchmark.get(scenario_name)
        print(f"\nmodel={monkey_model_name}, scenario={scenario_name}, benchmark={benchmark_name}")
        monkey_dataset = pd.read_json(path)
        
        monkey_questions2iscorrects = {row["question"]: row["is_corrects"] for _, row in monkey_dataset.iterrows()}
        print(f"len(monkey_questions2iscorrects): {len(monkey_questions2iscorrects)}")
        lengths = [len(l) for l in monkey_questions2iscorrects.values()]
        print(f"set(lengths): {set(lengths)}")

        if i == 0:
            pkl_name = "results_with_z" if benchmark_name != "safety" else "results_with_z_harmbench"
            with open(f"/lfs/skampere1/0/sttruong/deval/data/gather_helm_data/{pkl_name}.pkl", "rb") as f:
                helm_resmat_full = pickle.load(f)
            keep_cols = ~helm_resmat_full.columns.get_level_values("z").isna()
            helm_resmat_full = helm_resmat_full.loc[:, keep_cols]
            
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
        count = [
            (lst.index(1) + 1) if 1 in lst else (len(lst) + 1)
            for lst in monkey_questions2iscorrects.values()
        ]
        counts.append(count)
    
    K = torch.tensor(np.array(counts), dtype=torch.float64, device=device)
    K_max = K.max()
    n_llm, n_items = K.shape
    
    batch_theta_size = 150
    thetas_nuisance = torch.randn(batch_theta_size, n_llm, device=device)
    optimized_zs = []
    for i in tqdm(range(0, n_items, B)):
        end_idx = min(i + B, n_items)
        current_B = end_idx - i
        K_batch = K[:, i:end_idx]
        z_i = torch.randn(current_B, requires_grad=True, device=device)
        optim_z_i = LBFGS([z_i], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        def closure_z_i():
            optim_z_i.zero_grad()
            p_batch = torch.sigmoid(thetas_nuisance[:, :, None] + z_i[None, None, :])
            Kf = K_batch.float()
            logp = torch.log(p_batch + 1e-10)                 # ℓ1 = log(p)
            log1mp = torch.log(1.0 - p_batch + 1e-10)         # ℓ0 = log(1-p)
            mask_censored = (K_batch == (K_max + 1))
            ll_mat = torch.where(
                mask_censored,
                K_max * log1mp,                             # max cut: ℓ = K_max * log(1-p)
                (Kf - 1.0) * log1mp + logp                  # non max cut: ℓ = (K-1)*log(1-p) + log(p)
            )
            loss = -ll_mat.mean()
            loss.backward()
            return loss
        z_i_opt = trainer([z_i], optim_z_i, closure_z_i)[0].detach()
        optimized_zs.append(z_i_opt)
    zs = torch.cat(optimized_zs, dim=0)  # shape: (n_items,)

    thetas = torch.randn(n_llm, requires_grad=True, device=device)
    optim_theta = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_theta():
        optim_theta.zero_grad()
        p_mat = torch.sigmoid(thetas[:, None] + zs[None, :])
        Kf_all = K.float()                                  
        logp = torch.log(p_mat + 1e-10)                    
        log1mp = torch.log(1.0 - p_mat + 1e-10)            
        mask_cen_all = (Kf_all == (K_max + 1))
        ll_all = torch.where(
            mask_cen_all,
            K_max * log1mp,
            (Kf_all - 1.0) * log1mp + logp
        )
        loss = -ll_all.mean()
        loss.backward()
        return loss
    thetas = trainer([thetas], optim_theta, closure_theta)[0]
    
    print(pearsonr(thetas.detach().cpu().numpy(), K.mean(1).detach().cpu().numpy()))
    
    helm_zs = torch.tensor(helm_resmat.columns.get_level_values("z").astype(float), dtype=torch.float, device=device)
    
    print(pearsonr(helm_zs.detach().cpu().numpy(), zs.detach().cpu().numpy()))
    print(pearsonr(helm_zs.detach().cpu().numpy(), K.mean(0).detach().cpu().numpy()))
    print(pearsonr(zs.detach().cpu().numpy(), K.mean(0).detach().cpu().numpy()))
    
    plt.figure()
    plt.scatter(helm_zs.cpu().numpy(), zs.cpu().numpy(), marker='o', label='Train data')
    plt.xlabel('helm_zs')
    plt.ylabel('monkey_zs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"multiturn_helm_zs_vs_monkey_zs.png", dpi=300)