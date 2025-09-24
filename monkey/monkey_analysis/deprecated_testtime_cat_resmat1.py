import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
sys.path.append("../..")
from utils import cat_beta_1pl, compute_pass_datk_gts, compute_pass_datk_irt, cat_binary_1pl
from tueplots import bundles
bundles.icml2024()
from scipy.stats import spearmanr
from matplotlib import gridspec
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

if __name__ == "__main__":
    device = "cpu"
    sample_budget = 100
    # test llms: ['Qwen3-14B', 'Qwen3-32B', 'Qwen3-8B', 'gemma-3-27b-it']
    
    testtime_resmat1 = torch.load(f"irsl_testtime_resmat1_withz_betareg.pt", map_location="cpu")
    data_tensor = testtime_resmat1["data_tensor"].numpy()
    print(data_tensor.shape)
    model_names = testtime_resmat1["models"]
    datasets  = testtime_resmat1["datasets"]
    questions   = testtime_resmat1["questions"]
    zs = testtime_resmat1["zs"]
    helm_zs = testtime_resmat1["helm_zs"]
    
    results_dict = defaultdict(lambda: defaultdict(dict))
    for scen in tqdm(sorted(set(datasets))):
        output_dir = f"../../result/testtime_cat_resmat1/{scen}"
        os.makedirs(output_dir, exist_ok=True)

        idxs = [j for j, s in enumerate(datasets) if s == scen]
        sub_tensor = data_tensor[:, idxs, :]
        sub_zs = np.array(zs)[idxs]
        print(f"{scen}: shape = {sub_tensor.shape}")
        
        # z corr plot
        sub_helm_zs = np.array(helm_zs)[idxs]
        spearman_corr, _ = spearmanr(sub_zs, sub_helm_zs)
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            plt.figure(figsize=(6, 6))
            plt.scatter(sub_zs, sub_helm_zs, s=10, alpha=0.6)
            plt.xlabel("zs", fontsize=16)
            plt.ylabel("helm_zs", fontsize=16)
            plt.title(rf"{scen}: $\rho$={spearman_corr:.2f}", fontsize=20)
            plt.tick_params(axis="both", labelsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/zs_vs_helmzs.png", dpi=300, bbox_inches="tight")
            plt.close()
        
        sub_ys = torch.nanmean(
            torch.tensor(sub_tensor[:, :, :sample_budget], dtype=torch.float, device=device),
            dim = -1,
        )
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
        sub_binary = torch.tensor(sub_tensor[:, :, 0], dtype=torch.float, device=device)
        def _run_one_binary(i):
            return cat_binary_1pl(sub_binary[i], sub_zs, device)
        thetass_binary = Parallel(n_jobs=-1)(delayed(_run_one_binary)(i) for i in tqdm(range(sub_binary.shape[0])))
        thetass_binary = torch.tensor(thetass_binary, dtype=torch.float) # (n_models, budget)
        final_thetas_binary = thetass_binary[:, -1]
        
        # cat_beta_1pl(sub_ys[-1], sub_zs, device)
        def _run_one(i):
            return cat_beta_1pl(sub_ys[i], sub_zs, device)
        thetass = Parallel(n_jobs=-1)(delayed(_run_one)(i) for i in tqdm(range(sub_ys.shape[0])))
        thetass = torch.tensor(thetass, dtype=torch.float) # (n_models, budget)
        final_thetas = thetass[:, -1]
        
        # theta convergence
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(nrows=sub_ys.shape[0], ncols=1, figsize=(6, 2*sub_ys.shape[0]), sharex=True)
            for i, ax in enumerate(axes):
                ax.plot(np.arange(thetass.shape[-1]), thetass[i].cpu().numpy(), label=model_names[i])
                ax.set_ylabel("Theta", fontsize=16)
                ax.legend(fontsize=16)
                ax.tick_params(axis="both", labelsize=16)
            axes[-1].set_xlabel("Budget", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/theta_convergence.png", dpi=100, bbox_inches="tight")
            plt.close()
        
        for i, model in enumerate(model_names):
            model_data = sub_tensor[i]
            # plot of irt_corr_passat1s
            irt_probs = torch.sigmoid(float(final_thetas[i]) + sub_zs).cpu().numpy()
            irt_probs_binary = torch.sigmoid(float(final_thetas_binary[i]) + sub_zs).cpu().numpy()
            # irt_probs = sub_zs.cpu().numpy()
            passat1s = np.nanmean(model_data, axis=-1)
            rho, pval = spearmanr(irt_probs, passat1s, nan_policy="omit")
            results_dict[scen][model]["corr"] = rho
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig = plt.figure(figsize=(6, 6))
                gs = gridspec.GridSpec(
                    5, 5, figure=fig,
                    wspace=0.05, hspace=0.05
                )
                ax_scatter = fig.add_subplot(gs[0:4, 1:5])   # big square area
                ax_left    = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)  # LEFT marginal
                ax_bottom  = fig.add_subplot(gs[4,   1:5], sharex=ax_scatter)  # BOTTOM marginal

                # ---- scatter ----
                ax_scatter.scatter(irt_probs, passat1s, s=10)
                ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
                ax_scatter.set_xlim(0, 1)
                ax_scatter.set_ylim(0, 1)
                ax_scatter.set_xlabel(r"$\sigma(\theta+z)$", fontsize=18)
                ax_scatter.set_ylabel("pass@1", fontsize=18)
                ax_scatter.tick_params(axis="both", labelsize=14)
                
                # ---- left marginal (y) ----
                ax_left.hist(passat1s, bins=30, orientation="horizontal")
                ax_left.set_ylim(0, 1)
                ax_left.invert_xaxis()
                ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                ax_left.set_xticks([])
                for s in ("top", "right", "bottom", "left"):
                    ax_left.spines[s].set_visible(False)

                # ---- bottom marginal (x) ----
                ax_bottom.hist(irt_probs, bins=30)
                ax_bottom.set_xlim(0, 1)
                ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                ax_bottom.set_yticks([])
                for s in ("top", "right", "bottom", "left"):
                    ax_bottom.spines[s].set_visible(False)

                fig.suptitle(rf"{model}, {scen} ($\rho$ = {rho:.2f})", fontsize=16)
                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                fig.savefig(f"{output_dir}/irt_corr_passat1_{model}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
            
            # law curve before filter
            pass_datk_gts = compute_pass_datk_gts(model_data)
            passat1s_sub = np.nanmean(sub_tensor[i, :, :sample_budget], axis=-1)
            pass_datk_unbiased_subs = compute_pass_datk_irt(model_data, passat1s_sub)
            # pass_datk_gts = compute_pass_datk_irt(model_data, passat1s)
            pass_datk_irts = compute_pass_datk_irt(model_data, irt_probs)
            pass_datk_irts_binary = compute_pass_datk_irt(model_data, irt_probs_binary)
            
            mae = np.mean(np.abs(pass_datk_gts - pass_datk_irts))
            results_dict[scen][model]["mae_before_filter"] = mae
            n_samples = np.arange(1, model_data.shape[-1] + 1)
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                ax_left, ax_right = axes

                # Left: Pass@k vs k
                ax_left.plot(n_samples, pass_datk_gts, label="fullset unbiased (GT)", linewidth=2, color="blue")
                ax_left.plot(n_samples, pass_datk_unbiased_subs, label="subset unbiased", linestyle="--", linewidth=2, color="blue")
                ax_left.plot(n_samples, pass_datk_irts, label="subset beta-IRT", linewidth=2, color="red")
                ax_left.plot(n_samples, pass_datk_irts_binary, label="subset binary-IRT", linewidth=2, linestyle="--", color="red")
                ax_left.set_xlabel("Number of Samples", fontsize=16)
                ax_left.set_ylabel("Pass@k", fontsize=16)
                ax_left.set_ylim(0, 1)
                ax_left.legend(fontsize=14)
                ax_left.tick_params(axis="both", labelsize=14)

                # Right: log–log of -log(Pass@k) vs k
                ax_right.loglog(n_samples, -np.log(pass_datk_gts), label="fullset unbiased (GT)", linewidth=2, color="blue")
                ax_right.loglog(n_samples, -np.log(pass_datk_unbiased_subs), label="subset unbiased", linewidth=2, linestyle="--", color="blue")
                ax_right.loglog(n_samples, -np.log(pass_datk_irts), label="subset beta-IRT", linewidth=2, color="red")
                ax_right.loglog(n_samples, -np.log(pass_datk_irts_binary), label="subset binary-IRT", linewidth=2, linestyle="--", color="red")
                ax_right.set_xlabel("Number of Samples", fontsize=16)
                ax_right.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
                ax_right.legend(fontsize=14)
                ax_right.tick_params(axis="both", labelsize=14)

                fig.suptitle(rf"{model}, {scen} (Pass@k MAE = {mae:.2f})", fontsize=16)
                fig.tight_layout()
                fig.savefig(f"{output_dir}/law_curve_before_filter_{model}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                
            # law curve after filter
            # mask = (irt_probs >= 0.02)
            mask = (passat1s >= 0.01)
            model_data = model_data[mask]
            passat1s   = passat1s[mask]
            irt_probs  = irt_probs[mask]
            passat1s_sub = passat1s_sub[mask]

            pass_datk_gts = compute_pass_datk_irt(model_data, passat1s)
            pass_datk_unbiased_subs = compute_pass_datk_irt(model_data, passat1s_sub)
            pass_datk_irts = compute_pass_datk_irt(model_data, irt_probs)
            # pass_datk_gts = compute_pass_datk_irt(model_data, passat1s)
            pass_datk_irts_binary = compute_pass_datk_irt(model_data, irt_probs_binary)
            
            mae = np.mean(np.abs(pass_datk_gts - pass_datk_irts))
            results_dict[scen][model]["mae_after_filter"] = mae
            n_samples = np.arange(1, model_data.shape[-1] + 1)
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                ax_left, ax_right = axes

                # Left: Pass@k vs k
                ax_left.plot(n_samples, pass_datk_gts, label="fullset unbiased (GT)", linewidth=2, color="blue")
                ax_left.plot(n_samples, pass_datk_unbiased_subs, label="subset unbiased", linestyle="--", linewidth=2, color="blue")
                ax_left.plot(n_samples, pass_datk_irts, label="subset beta-IRT", linewidth=2, color="red")
                ax_left.plot(n_samples, pass_datk_irts_binary, label="subset binary-IRT", linewidth=2, linestyle="--", color="red")
                ax_left.set_xlabel("Number of Samples", fontsize=16)
                ax_left.set_ylabel("Pass@k", fontsize=16)
                ax_left.set_ylim(0, 1)
                ax_left.legend(fontsize=14)
                ax_left.tick_params(axis="both", labelsize=14)

                # Right: log–log of -log(Pass@k) vs k
                ax_right.loglog(n_samples, -np.log(pass_datk_gts), label="fullset unbiased (GT)", linewidth=2, color="blue")
                ax_right.loglog(n_samples, -np.log(pass_datk_unbiased_subs), label="subset unbiased", linewidth=2, linestyle="--", color="blue")
                ax_right.loglog(n_samples, -np.log(pass_datk_irts), label="subset beta-IRT", linewidth=2, color="red")
                ax_right.loglog(n_samples, -np.log(pass_datk_irts_binary), label="subset binary-IRT", linewidth=2, linestyle="--", color="red")
                ax_right.set_xlabel("Number of Samples", fontsize=16)
                ax_right.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
                ax_right.legend(fontsize=14)
                ax_right.tick_params(axis="both", labelsize=14)

                fig.suptitle(rf"{model}, {scen} (Pass@k MAE = {mae:.2f})", fontsize=16)
                fig.tight_layout()
                fig.savefig(f"{output_dir}/law_curve_after_filter{model}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
        
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"testtime_resmat1_results.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)