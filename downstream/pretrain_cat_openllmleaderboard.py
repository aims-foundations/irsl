import pickle
from datasets import load_dataset
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
torch.manual_seed(0)
torch.set_num_threads(1)
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.append("..")
import os
from utils import beta_nll
from tueplots import bundles
bundles.icml2024()

def translate_str(s): # e.g., "300B", "64M"
    if s.endswith("M"):
        return float(s[:-1]) * 1e6
    elif s.endswith("B"):
        return float(s[:-1]) * 1e9
    elif s.endswith("T"):
        return float(s[:-1]) * 1e12
    else:
        raise ValueError(f"Unrecognized size format in: {s}")

def calculate_flop(s):
    traindata_size = translate_str(s.split("_")[-1])
    model_size = translate_str(s.split("_")[-2])
    return traindata_size * model_size
    # return traindata_size * model_size / 1e21
    
def trainer(parameters, optim, closure, n_iter=100, verbose=False, eps=1e-6):
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
            
            if d_loss < eps and d_parameters < eps and grad_norm < eps:
                break
            
    return parameters

def estimate_theta_beta(theta, asked_ys, asked_discris, asked_zs, device, phi=10.0, eps=1e-6, lr=0.1):
    asked_ys = torch.as_tensor(asked_ys, device=device, dtype=torch.float)
    asked_discris = torch.as_tensor(asked_discris, device=device, dtype=torch.float)
    asked_zs = torch.as_tensor(asked_zs, device=device, dtype=torch.float)
    asked_ys = asked_ys.clamp(min=eps, max=1.0 - eps)
    theta = theta.clone().requires_grad_(True)
    optim = torch.optim.LBFGS([theta], lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    phi_t = torch.as_tensor(phi, device=device, dtype=torch.float)
    def closure():
        optim.zero_grad()
        mu = torch.sigmoid(asked_discris*(theta - asked_zs))
        mu = mu.clamp(min=eps, max=1.0 - eps)
        loss = beta_nll(asked_ys, mu, phi_t).mean()
        loss.backward()
        return loss
    theta = trainer([theta], optim, closure)[0]
    return theta.detach()

def compute_fisher_info(theta, remain_discris, remain_zs):
    p = torch.sigmoid(remain_discris*(theta[:, None] - remain_zs[None, :]))
    return p * (1 - p)

def cat_beta(ys, discris, zs, device, budget):
    adaptive_theta_hat = torch.zeros((1,), device=device)
    adaptive_theta_hats = [adaptive_theta_hat]
    adaptive_asked_discris = []
    adaptive_asked_zs = []
    adaptive_asked_ys = []
    remain_discris = discris.clone()
    remain_zs = zs.clone()
    remain_ys = ys.clone()
    
    asked = 0
    while asked < budget and remain_ys.numel() > 0:
        fisher_info = compute_fisher_info(adaptive_theta_hat, remain_discris, remain_zs).squeeze(0)
        next_item = torch.argmax(fisher_info)
        y_i = remain_ys[next_item]
        a_i = remain_discris[next_item]
        z_i = remain_zs[next_item]
        remain_discris = torch.cat([remain_discris[:next_item], remain_discris[next_item + 1:]])
        remain_zs = torch.cat([remain_zs[:next_item], remain_zs[next_item + 1:]])
        remain_ys = torch.cat([remain_ys[:next_item], remain_ys[next_item + 1:]])
        if torch.isnan(y_i):
            continue
        adaptive_asked_discris.append(a_i)
        adaptive_asked_zs.append(z_i)
        adaptive_asked_ys.append(y_i)
        adaptive_theta_hat = estimate_theta_beta(adaptive_theta_hat, adaptive_asked_ys, adaptive_asked_discris, adaptive_asked_zs, device)
        adaptive_theta_hats.append(adaptive_theta_hat)
        asked += 1
    return torch.tensor(adaptive_theta_hats, dtype=torch.float, device=device)

DATASETS = ["mmlu_anatomy", "arc_challenge", "hellaswag", "mmlu_abstract_algebra", "mmlu_astronomy"]

if __name__ == "__main__":
    device = "cpu"
    budget = 200
    results_dict = defaultdict(lambda: defaultdict(dict))
    for dataset in DATASETS:
        dataset_temp = "mmlu" if dataset.startswith("mmlu") else dataset
        ds = load_dataset(f"RylanSchaeffer/per_sample_scores.{dataset}.prob_choices_correct.2024-05-31")
        df = ds["train"].to_pandas()
        model_nicknames = df["Model Nickname"].unique()
        model_families = pd.Series(model_nicknames).str.split("_").str[0].unique()
        
        for model_family in model_families:
            output_dir = f"../result/pretrain_betacat_openllmleaderboard/{dataset}_{model_family}"
            os.makedirs(output_dir, exist_ok=True)

            df_f = df[df["Model Nickname"].astype(str).str.startswith(model_family)].copy()
            resmat = df_f.pivot_table(
                index="Model Nickname",
                columns="sample_idx",
                values="score",
            )
            resmat = resmat.sort_index(axis=1)
            resmat = resmat.loc[sorted(resmat.index, key=lambda x: calculate_flop(x))]
            flops = [calculate_flop(model_name)for model_name in resmat.index]
            
            cache_dir = snapshot_download(repo_id="allenai/fluid-benchmarking", repo_type="dataset")
            irt_df = pd.read_csv(f"{cache_dir}/data/irt_models/{dataset_temp}.csv", names=["raw_idx", "a", "b"], header=0)
            col_with_dataset = [f"{dataset}_{c}" for c in resmat.columns]
            irt_df = irt_df.set_index("raw_idx").loc[col_with_dataset]
            discris, zs = irt_df["a"].values, irt_df["b"].values
            assert discris.shape[0] == zs.shape[0] == resmat.shape[1]
            
            zs = torch.tensor(zs, dtype=torch.float, device=device)
            discris = torch.tensor(discris, dtype=torch.float, device=device)
            ys = torch.tensor(resmat.values, dtype=torch.float, device=device)
            
            # cat_beta(ys[-1], discris, zs, device, budget)
            def _run_one(i):
                return cat_beta(ys[i], discris, zs, device, budget)
            thetass = Parallel(n_jobs=-1)(delayed(_run_one)(i) for i in tqdm(range(ys.shape[0])))
            thetass = torch.stack(thetass) # (n_models, budget)
            final_thetas = thetass[:, -1]
            
            # theta convergence
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, axes = plt.subplots(nrows=ys.shape[0], ncols=1, figsize=(6, 2*ys.shape[0]), sharex=True)
                budgets = np.arange(budget+1)
                for i, ax in enumerate(axes):
                    ax.plot(budgets, thetass[i].cpu().numpy(), label=f"{flops[i]/1e21:.2f} * 1e21")
                    ax.set_ylabel("Theta", fontsize=16)
                    ax.legend(fontsize=16)
                    ax.tick_params(axis="both", labelsize=16)
                axes[-1].set_xlabel("Budget", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/theta_convergence.png", dpi=100, bbox_inches="tight")
                plt.close()
            
            # law curve
            final_thetas = thetass[:, -1].cpu().numpy()
            means_all = torch.nanmean(ys, dim=1).cpu().numpy() # mean score on all questions
            means_sub = torch.nanmean(ys[:, torch.randperm(ys.shape[1])[:budget]], dim=1).cpu().numpy() # mean score on random subset
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax1 = plt.subplots(figsize=(6, 4))
                ax1.plot(flops, means_sub, color="tab:blue", linewidth=1.5, label="random subset mean")
                ax1.plot(flops, means_all, color='tab:blue', linewidth=1.5, label="full set mean", linestyle="--")
                ax1.set_xlabel("FLOP", fontsize=16)
                ax1.set_xscale("log")
                ax1.set_ylabel("Mean score", color='tab:blue', fontsize=16)
                ax1.tick_params(axis="x", labelsize=16)
                ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)
                ax2 = ax1.twinx()
                ax2.plot(flops, final_thetas, color='tab:red', linewidth=1.5, label=r"CAT $\theta$")
                ax2.set_ylabel(r"CAT $\theta$", color='tab:red', fontsize=16)
                # ax2.set_ylim(-5, None)
                ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=16)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/law_curve.png", dpi=300, bbox_inches="tight")
                plt.close()
                
            results_dict[dataset][model_family] = {
                    "flops": flops,
                    "resmat": resmat,
                    "thetass": thetass,
                }
        
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"../result/pretrain_cat/result.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)