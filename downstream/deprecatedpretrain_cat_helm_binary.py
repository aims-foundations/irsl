import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
import sys
sys.path.append("..")
from utils import visualize_response_matrix
from tueplots import bundles
bundles.icml2024()
from huggingface_hub import snapshot_download
from torch.distributions import Bernoulli
import logging

def estimate_theta(theta, asked_ys, asked_zs, device):
    def closure():
        optim.zero_grad()
        probs = torch.sigmoid(theta[:, None] + asked_zs[None, :])
        loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        loss.backward()
        return loss

    asked_ys = torch.tensor(asked_ys, device=device)
    asked_zs = torch.tensor(asked_zs, device=device)
    theta = theta.clone().requires_grad_(True)
    optim = torch.optim.LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
    for iteration in range(100):
        if iteration > 0:
            previous_theta = theta.clone()
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = previous_loss - loss
            d_theta = torch.norm(previous_theta - theta, p=2)
            grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
            if d_loss < 1e-5 and d_theta < 1e-5 and grad_norm < 1e-5:
                break
    
    return theta.detach()

# def compute_fisher_info(theta, remain_zs):
#     p = torch.sigmoid(theta[:, None] + remain_zs[None, :])
#     return p * (1 - p)

def cat(ys, zs, device, budget):
    adaptive_theta_hat = torch.zeros((1,), device=device)
    adaptive_theta_hats = [adaptive_theta_hat]
    adaptive_asked_zs = []
    adaptive_asked_ys = []
    remain_zs = zs.clone()
    remain_ys = ys.clone()
    
    asked = 0
    while asked < budget and remain_zs.numel() > 0:
        next_item = torch.argmin(abs(adaptive_theta_hat + remain_zs))
        y_val = remain_ys[next_item]
        z_val = remain_zs[next_item]
        remain_zs = torch.cat([remain_zs[:next_item], remain_zs[next_item + 1:]])
        remain_ys = torch.cat([remain_ys[:next_item], remain_ys[next_item + 1:]])
        if torch.isnan(y_val):
            continue
        adaptive_asked_ys.append(y_val)
        adaptive_asked_zs.append(z_val)
        adaptive_theta_hat = estimate_theta(
            adaptive_theta_hat, adaptive_asked_ys, adaptive_asked_zs, device
        )
        adaptive_theta_hats.append(adaptive_theta_hat)
        asked += 1
    return torch.tensor(adaptive_theta_hats, dtype=torch.float, device=device)

REPO_IDS = [
        "EleutherAI/pythia-12b",
        "EleutherAI/pythia-6.9b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-14m",
        "LLM360/Amber",
        "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints",
        "HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints",
        "HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints",
    ]
MODELS = [repo_id.split("/")[1] for repo_id in REPO_IDS]

SCENARIOS = ['babi_qa', 'civil_comments', 'commonsense',
    'dyck_language_np=3', 'entity_data_imputation', 'entity_matching',
    'gsm', 'legal_support', 'legalbench', 'mmlu', 'raft',
    'synthetic_reasoning', 'wikifact'] # 'med_qa', 'boolq', 'imdb'

if __name__ == "__main__":
    device = "cpu" # "cuda:4"
    budget = 100
    max_workers = 256
    results_dict = defaultdict(lambda: defaultdict(dict))
    
    os.makedirs("../result/pretrain_cat", exist_ok=True)
    logging.basicConfig(
        filename="../result/pretrain_cat/run.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    cache_dir = snapshot_download(repo_id="stair-lab/irsl_downstream_resmat", repo_type="dataset")
    for model in tqdm(MODELS):
        with open(f"{cache_dir}/results_{model}.pkl", "rb") as f:
            resmat_full = pickle.load(f)
        
        for scenario in tqdm(SCENARIOS):
            output_dir = f"../result/pretrain_cat/{scenario}_{model}"
            os.makedirs(output_dir, exist_ok=True)
            resmat = resmat_full.loc[:, ~resmat_full.columns.get_level_values("z").isna()]
            resmat = resmat.loc[:, resmat.columns.get_level_values("scenario") == scenario]
            resmat = resmat[~resmat.isna().all(axis=1)]
            visualize_response_matrix(resmat, resmat, f"{output_dir}/response_matrix.png")
            
            steps = np.array([float(name.split("-")[-1]) for name in resmat.index])
            ys = torch.tensor(resmat.values, dtype=torch.float, device=device)
            n_test_takers, n_items = ys.shape
            nan_pct = torch.isnan(ys).float().mean().item() * 100
            logging.info(f"model={model} scenario={scenario} shape={ys.shape} nan_pct={nan_pct:.2f}%")
            zs = torch.tensor(resmat.columns.get_level_values("z").astype(float), dtype=torch.float, device=device)
            
            # z distribution
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figsize=(6, 6)
                plt.hist(zs.cpu().numpy(), bins=30)
                plt.xlabel("z values", fontsize=10)
                plt.ylabel("Frequency", fontsize=10)
                plt.tick_params(axis="both", labelsize=10)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/zs_distribution.png", dpi=300, bbox_inches="tight")
                plt.close()

            # irt theta on subset questions
            def _run_one(i):
                return cat(ys[i], zs, device, budget)
            thetass = Parallel(n_jobs=max_workers)(delayed(_run_one)(i) for i in tqdm(range(ys.shape[0])))
            thetass = torch.stack(thetass)
            
            # theta convergence
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, axes = plt.subplots(nrows=ys.shape[0], ncols=1, figsize=(6, 2*ys.shape[0]), sharex=True)
                budgets = np.arange(budget+1)
                for i, ax in enumerate(axes):
                    ax.plot(budgets, thetass[i].cpu().numpy(), label=int(steps[i]))
                    ax.set_ylabel("Theta", fontsize=16)
                    ax.legend(fontsize=16)
                    ax.tick_params(axis="both", labelsize=16)
                axes[-1].set_xlabel("Budget", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/theta_convergence.png", dpi=100, bbox_inches="tight")
                plt.close()
            
            # law curve
            final_thetas = thetass[:, -1].cpu().numpy()
            step_pcts = steps.astype(int) / steps.astype(int).max() * 100.0
            means_all = torch.nanmean(ys, dim=1).cpu().numpy() # mean score on all questions
            means_sub = torch.nanmean(ys[:, torch.randperm(ys.shape[1])[:budget]], dim=1).cpu().numpy() # mean score on random subset
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax1 = plt.subplots(figsize=(6, 4))
                ax1.plot(step_pcts, means_sub, color="tab:blue", linewidth=1.5, label="random subset mean")
                ax1.plot(step_pcts, means_all, color='tab:blue', linewidth=1.5, label="full set mean", linestyle="--")
                ax1.set_xlabel(r"Training progress (\%)", fontsize=16)
                ax1.set_ylabel("Mean score", color='tab:blue', fontsize=16)
                ax1.tick_params(axis="x", labelsize=16)
                ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)
                ax2 = ax1.twinx()
                ax2.plot(step_pcts, final_thetas, color='tab:red', linewidth=1.5, label=r"CAT $\theta$")
                ax2.set_ylabel(r"CAT $\theta$", color='tab:red', fontsize=16)
                ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=16)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/law_curve.png", dpi=300, bbox_inches="tight")
                plt.close()
            
            results_dict[scenario][model] = {
                "steps": steps,
                "ys": ys,
                "thetass": thetass,
            }
        
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"../result/pretrain_cat/result.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)