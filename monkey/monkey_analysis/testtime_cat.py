import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
sys.path.append("../..")
from utils import visualize_response_matrix, cat
from tueplots import bundles
bundles.icml2024()
from huggingface_hub import snapshot_download
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

def compute_pass_iatk_gt(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_pass_datk_gts(data2d: np.ndarray) -> np.ndarray:
    n_items, n_samples = data2d.shape
    k_range = np.arange(1, n_samples + 1)
    per_item = []
    for i in range(n_items):
        arr = data2d[i]
        valid = ~np.isnan(arr) # if data2d is torch.tensor, this line return a list of 255
        n = int(valid.sum())
        c = int(np.nansum(arr))
        per_item.append([
            compute_pass_iatk_gt(n, c, k)
            for k in k_range
        ])
    return np.nanmean(np.vstack(per_item), axis=0)

def compute_pass_iatk_irt(pass_iat1: float, k: int) -> float:
    return 1.0 - (1.0 - pass_iat1) ** k

def compute_pass_datk_irt(data2d: np.ndarray, irt_probs: np.ndarray) -> np.ndarray:
    n_items, n_samples = data2d.shape
    assert n_items == irt_probs.shape[0]
    k_range = np.arange(1, n_samples + 1)
    per_item = []
    for i in range(n_items):
        per_item.append([
            compute_pass_iatk_irt(irt_probs[i], k)
            for k in k_range
        ])
    return np.nanmean(np.vstack(per_item), axis=0)

if __name__ == "__main__":
    device = "cpu" # "cuda:4"
    budget = 30
    max_workers = 16
    
    # cache_dir = snapshot_download(repo_id="stair-lab/irsl_testtime_resmat2", repo_type="dataset")
    testtime_resmat2 = torch.load(f"irsl_testtime_resmat2_withz.pt", map_location="cpu", weights_only=False)
    data_tensor = testtime_resmat2["data_tensor"].numpy()
    print(data_tensor.shape)
    model_names = testtime_resmat2["models"]
    datasets  = testtime_resmat2["datasets"]
    questions   = testtime_resmat2["questions"]
    zs = testtime_resmat2["zs"]
    
    for scen in tqdm(set(datasets)):
        output_dir = f"../../result/testtime_cat/{scen}"
        os.makedirs(output_dir, exist_ok=True)

        idxs = [j for j, s in enumerate(datasets) if s == scen]
        sub_tensor = data_tensor[:, idxs, :]
        sub_zs = np.array(zs)[idxs]
        print(f"{scen}: shape = {sub_tensor.shape}")
        
        sub_ys = sub_tensor[:, :, 0]
        sub_ys = torch.tensor(sub_ys, dtype=torch.float, device=device)
        n_test_takers, n_items = sub_ys.shape
        sub_zs = torch.tensor(sub_zs, dtype=torch.float, device=device)
        
        # z distribution
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            plt.figsize=(6, 6)
            plt.hist(sub_zs.cpu().numpy(), bins=30)
            plt.xlabel("z values", fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.tick_params(axis="both", labelsize=10)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/zs_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
        
        # irt theta on subset questions
        def _run_one(i):
            return cat(sub_ys[i], sub_zs, device, budget)
        thetass = Parallel(n_jobs=max_workers)(delayed(_run_one)(i) for i in tqdm(range(sub_ys.shape[0])))
        thetass = torch.stack(thetass) # (n_models, budget)
        final_thetas = thetass[:, -1]
        
        # theta convergence
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(nrows=sub_ys.shape[0], ncols=1, figsize=(6, 2*sub_ys.shape[0]), sharex=True)
            budgets = np.arange(budget+1)
            for i, ax in enumerate(axes):
                ax.plot(budgets, thetass[i].cpu().numpy(), label=model_names[i])
                ax.set_ylabel("Theta", fontsize=16)
                ax.legend(fontsize=16)
                ax.tick_params(axis="both", labelsize=16)
            axes[-1].set_xlabel("Budget", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/theta_convergence.png", dpi=100, bbox_inches="tight")
            plt.close()
        
        for i, model in enumerate(model_names):
            model_data = sub_tensor[i]
            # plot of irt_corr_passat1
            irt_probs = torch.sigmoid(float(final_thetas[i]) + sub_zs).cpu().numpy()
            passat1 = np.nanmean(model_data, axis=-1)
            rho, pval = spearmanr(irt_probs, passat1, nan_policy="omit")
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(6, 6))
                plt.scatter(irt_probs, passat1, s=10)
                plt.xlabel(r"$\sigma$($\theta$ + $z$)", fontsize=18)
                plt.ylabel("pass@1", fontsize=18)
                plt.xlim(0,1)
                plt.ylim(0,1)
                plt.title(rf"{model, scen} ($\rho$ = {rho:.2f})", fontsize=16)
                plt.tick_params(axis="both", labelsize=14)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/irt_corr_passat1_{model}.png", dpi=300, bbox_inches="tight")
                plt.close()
            
            # law curve
            pass_datk_gts = compute_pass_datk_gts(model_data)
            pass_datk_irts = compute_pass_datk_irt(model_data, irt_probs)
            mae = np.mean(np.abs(pass_datk_gts - pass_datk_irts))
            n_samples = np.arange(1, model_data.shape[-1] + 1)
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(6, 6))
                plt.plot(n_samples, pass_datk_gts, label="GT", linewidth=2)
                plt.plot(n_samples, pass_datk_irts, label="IRT Est", linewidth=2, linestyle="--")
                plt.xlabel("Number of Samples", fontsize=16)
                plt.ylabel("Pass@k", fontsize=16)
                plt.title(rf"{model}, {scen} (MAE = {mae:.2f})", fontsize=16)
                plt.legend(fontsize=14)
                plt.ylim(0,1)
                plt.tick_params(axis="both", labelsize=14)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/law_curve_{model}.png", dpi=300, bbox_inches="tight")
                plt.close()