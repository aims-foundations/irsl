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
sys.path.append("../..")
from utils import visualize_response_matrix, cat
from tueplots import bundles
bundles.icml2024()
from huggingface_hub import snapshot_download
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    device = "cpu" # "cuda:4"
    budget = 30
    max_workers = 16
    
    cache_dir = snapshot_download(repo_id="stair-lab/irsl_testtime_resmat1", repo_type="dataset")
    testtime_resmat1 = torch.load(f"{cache_dir}/testtime_resmat2.pt", map_location="cpu", weights_only=False)
    data_tensor = testtime_resmat1["data_tensor"]
    print(data_tensor.shape)
    model_names = testtime_resmat1["models"]
    benchmarks  = testtime_resmat1["benchmarks"]
    scenarios   = testtime_resmat1["scenarios"]
    questions   = testtime_resmat1["questions"]
    zs = testtime_resmat1["zs"]
    
    results_dict = defaultdict(lambda: defaultdict(dict))
    for scen in tqdm(set(scenarios)):
        output_dir = f"../../result/testtime_cat/{scen}"
        os.makedirs(output_dir, exist_ok=True)
            
        idxs = [j for j, s in enumerate(scenarios) if s == scen]
        sub_tensor = data_tensor[:, idxs, :]
        print(f"{scen}: shape = {sub_tensor.shape}")
        resmat = sub_tensor[:, :, 0]
        # visualize_response_matrix(resmat, resmat, f"{output_dir}/response_matrix.png")
        sub_ys = torch.tensor(resmat, dtype=torch.float, device=device)
        n_test_takers, n_items = sub_ys.shape
        sub_zs = torch.tensor(np.array(zs)[idxs], dtype=torch.float, device=device)
        
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
            # plot of irt_corr_passat1
            irt_prob = torch.sigmoid(float(final_thetas[i]) + sub_zs).cpu().numpy()
            # irt_prob = (float(final_thetas[i]) + sub_zs).cpu().numpy()
            # irt_prob = sub_zs.cpu().numpy()
            passat1 = sub_tensor.mean(-1)[i]
            rho, pval = spearmanr(irt_prob, passat1, nan_policy="omit")
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(6, 6))
                plt.scatter(irt_prob, passat1, s=10)
                plt.xlabel(r"$\theta$ + $z$", fontsize=18)
                plt.ylabel("pass@1", fontsize=18)
                plt.title(rf"{model, scen} ($\rho$ = {rho:.2f})", fontsize=16)
                plt.tick_params(axis="both", labelsize=14)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/irt_corr_passat1_{model}.png", dpi=300, bbox_inches="tight")
                plt.close()
            
            # law curve
            
    #     step_pcts = steps.astype(int) / steps.astype(int).max() * 100.0
    #     means_all = torch.nanmean(sub_ys, dim=1).cpu().numpy() # mean score on all questions
    #     means_sub = torch.nanmean(sub_ys[:, torch.randperm(sub_ys.shape[1])[:budget]], dim=1).cpu().numpy() # mean score on random subset
    #     with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    #         fig, ax1 = plt.subplots(figsize=(6, 4))
    #         ax1.plot(step_pcts, means_sub, color="tab:blue", linewidth=1.5, label="random subset mean")
    #         ax1.plot(step_pcts, means_all, color='tab:blue', linewidth=1.5, label="full set mean", linestyle="--")
    #         ax1.set_xlabel(r"Training progress (\%)", fontsize=16)
    #         ax1.set_ylabel("Mean score", color='tab:blue', fontsize=16)
    #         ax1.tick_params(axis="x", labelsize=16)
    #         ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)
    #         ax2 = ax1.twinx()
    #         ax2.plot(step_pcts, final_thetas, color='tab:red', linewidth=1.5, label=r"CAT $\theta$")
    #         ax2.set_ylabel(r"CAT $\theta$", color='tab:red', fontsize=16)
    #         ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=16)
    #         lines1, labels1 = ax1.get_legend_handles_labels()
    #         lines2, labels2 = ax2.get_legend_handles_labels()
    #         ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=16)
    #         plt.tight_layout()
    #         plt.savefig(f"{output_dir}/law_curve.png", dpi=300, bbox_inches="tight")
    #         plt.close()

                    
    #     results_dict[scenario][model] = {
    #         "steps": steps,
    #         "sub_ys": sub_ys,
    #         "thetass": thetass,
    #     }
    
    # final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    # with open(f"{output_dir}/result.pkl", "wb") as f:
    # pickle.dump(final_results_dict, f)