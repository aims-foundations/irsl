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

def estimate_theta_beta(theta, asked_ys, asked_zs, device, phi=10.0, eps=1e-6, lr=0.1):
    asked_ys = torch.as_tensor(asked_ys, device=device, dtype=torch.float)
    asked_zs = torch.as_tensor(asked_zs, device=device, dtype=torch.float)
    asked_ys = asked_ys.clamp(min=eps, max=1.0 - eps)
    theta = theta.clone().requires_grad_(True)
    optim = torch.optim.LBFGS([theta], lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    phi_t = torch.as_tensor(phi, device=device, dtype=torch.float)
    def closure():
        optim.zero_grad()
        mu = torch.sigmoid(theta[:, None] + asked_zs[None, :])
        mu = mu.clamp(min=eps, max=1.0 - eps)
        loss = beta_nll(asked_ys, mu, phi_t).mean()
        loss.backward()
        return loss
    theta = trainer([theta], optim, closure)[0]
    return theta.detach()

def cat_beta(ys, zs, device, budget):
    adaptive_theta_hat = torch.zeros((1,), device=device)
    adaptive_theta_hats = [adaptive_theta_hat]
    adaptive_asked_zs = []
    adaptive_asked_ys = []
    remain_zs = zs.clone()
    remain_ys = ys.clone()
    
    asked = 0
    while asked < budget and remain_ys.numel() > 0:
        next_item = torch.argmin(abs(adaptive_theta_hat + remain_zs))
        y_i = remain_ys[next_item]
        z_i = remain_zs[next_item]
        remain_zs = torch.cat([remain_zs[:next_item], remain_zs[next_item + 1:]])
        remain_ys = torch.cat([remain_ys[:next_item], remain_ys[next_item + 1:]])
        if torch.isnan(y_i):
            continue
        adaptive_asked_zs.append(z_i)
        adaptive_asked_ys.append(y_i)
        adaptive_theta_hat = estimate_theta_beta(adaptive_theta_hat, adaptive_asked_ys, adaptive_asked_zs, device)
        adaptive_theta_hats.append(adaptive_theta_hat)
        asked += 1
    return torch.tensor(adaptive_theta_hats, dtype=torch.float, device=device)

REPO_IDS = [
        # "EleutherAI/pythia-12b",
        # "EleutherAI/pythia-6.9b",
        # "EleutherAI/pythia-2.8b",
        # "EleutherAI/pythia-1.4b",
        # "EleutherAI/pythia-1b",
        # "EleutherAI/pythia-410m",
        # "EleutherAI/pythia-160m",
        # "EleutherAI/pythia-70m",
        # "EleutherAI/pythia-14m",
        "LLM360/Amber",
        # "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints",
        # "HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints",
        # "HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints",
    ]
MODELS = [repo_id.split("/")[1] for repo_id in REPO_IDS]

SCENARIOS = ['civil_comments', 'entity_matching', 'legal_support', 'raft', 'boolq', 'imdb']

STEP2FLOP = {
    "pythia": 300 * 1e9 / 143000,
    "Amber": 1.26 * 1e12 / 358,
    "SmolLM2-1.7B": 250 * 1e9 / 125000,
    "SmolLM2-360M": 250 * 1e9 / 160000,
    "SmolLM2-135M": 250 * 1e9 / 240000,
}

def calculate_flop(step: float, model: str) -> float:
    for key, factor in STEP2FLOP.items():
        if model.startswith(key):
            return step * factor
    raise ValueError(f"Unknown model family for '{model}'")

if __name__ == "__main__":
    device = "cpu"
    budget = 100
    results_dict = defaultdict(lambda: defaultdict(dict))
    
    log_dir = "../result/pretrain_betacat_helm"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/run.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    cache_dir = snapshot_download(repo_id="stair-lab/irsl_downstream_resmat1", repo_type="dataset")
    for model in tqdm(MODELS):
        # with open(f"{cache_dir}/resmat/results_{model}.pkl", "rb") as f:
        with open(f"/lfs/skampere2/0/sttruong/irsl/data/pretrain_helm/results_Amber_sttruong_skampere2.pkl", "rb") as f:
            resmat_full = pickle.load(f)
        
        for scenario in tqdm(SCENARIOS):
            output_dir = f"../result/pretrain_betacat_helm/{scenario}_{model}"
            os.makedirs(output_dir, exist_ok=True)
            resmat = resmat_full.loc[:, ~resmat_full.columns.get_level_values("z").isna()]
            resmat = resmat.loc[:, resmat.columns.get_level_values("scenario") == scenario]
            resmat = resmat[~resmat.isna().all(axis=1)]
            visualize_response_matrix(resmat, resmat, f"{output_dir}/response_matrix.png")
            
            steps = np.array([float(name.split("-")[-1]) for name in resmat.index])
            flops = np.array([calculate_flop(step, model) for step in steps])
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
                return cat_beta(ys[i], zs, device, budget)
            thetass = Parallel(n_jobs=-1)(delayed(_run_one)(i) for i in tqdm(range(ys.shape[0])))
            thetass = torch.stack(thetass)
            
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
                ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=16)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/law_curve.png", dpi=300, bbox_inches="tight")
                plt.close()
            
            results_dict[scenario][model] = {
                "flops": flops,
                "resmat": resmat,
                "thetass": thetass,
            }
        
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"../result/pretrain_betacat_helm/result.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)