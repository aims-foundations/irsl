import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from tqdm import tqdm
from joblib import Parallel, delayed
from testtime_calibrate_resmat1_beta_regr import beta_nll
from tueplots import bundles
bundles.icml2024()
from huggingface_hub import snapshot_download
from scipy.stats import spearmanr
from matplotlib import gridspec
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

def estimate_theta_beta(theta, asked_ys, asked_zs, device, phi=10.0, eps=1e-6):
    asked_ys = torch.as_tensor(asked_ys, device=device, dtype=torch.float)
    asked_zs = torch.as_tensor(asked_zs, device=device, dtype=torch.float)
    asked_ys = asked_ys.clamp(min=eps, max=1.0 - eps)
    theta = theta.clone().requires_grad_(True)
    optim = torch.optim.LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    phi_t = torch.as_tensor(phi, device=device, dtype=torch.float)
    def closure():
        optim.zero_grad()
        mu = torch.sigmoid(theta + asked_zs)
        mu = mu.clamp(min=eps, max=1.0 - eps)
        loss = beta_nll(asked_ys, mu, phi_t).mean()
        loss.backward()
        return loss
    prev_loss = None
    for _ in range(100):
        loss = optim.step(closure)
        if prev_loss is not None and abs(prev_loss.item() - loss.item()) < 1e-5:
            break
        prev_loss = loss
    return theta.detach()

# def compute_fisher_info(theta, remain_zs):
#     p = torch.sigmoid(theta[:, None] + remain_zs[None, :])
#     return p * (1 - p)

def cat_beta(ys, zs, device, budget):
    adaptive_theta_hat = torch.zeros((1,), device=device)
    adaptive_theta_hats = [adaptive_theta_hat]
    adaptive_asked_zs = []
    adaptive_asked_ys = []
    remain_zs = zs.clone()
    remain_ys = ys.clone()
    for _ in range(budget):
        # fisher_info = compute_fisher_info(adaptive_theta_hat, remain_zs)
        # next_item = torch.argmax(fisher_info)
        next_item = torch.argmin(abs(adaptive_theta_hat + remain_zs))
        adaptive_asked_zs.append(remain_zs[next_item])
        adaptive_asked_ys.append(remain_ys[next_item])
        adaptive_theta_hat = estimate_theta_beta(adaptive_theta_hat, adaptive_asked_ys, adaptive_asked_zs, device)
        adaptive_theta_hats.append(adaptive_theta_hat)
        remain_zs = torch.cat([remain_zs[:next_item], remain_zs[next_item + 1:]])
        remain_ys = torch.cat([remain_ys[:next_item], remain_ys[next_item + 1:]])
    
    return torch.tensor(adaptive_theta_hats, dtype=torch.float, device=device)

if __name__ == "__main__":
    device = "cpu"
    item_budget = 70
    sample_budget = 100
    max_workers = 16
    # test llms: ['Qwen3-14B', 'Qwen3-32B', 'Qwen3-8B', 'gemma-3-27b-it']
    
    testtime_resmat1 = torch.load(f"irsl_testtime_resmat1_withz_betareg.pt", map_location="cpu", weights_only=False)
    data_tensor = testtime_resmat1["data_tensor"].numpy()
    print(data_tensor.shape)
    model_names = testtime_resmat1["models"]
    datasets  = testtime_resmat1["datasets"]
    questions   = testtime_resmat1["questions"]
    zs = testtime_resmat1["zs"]
    helm_zs = testtime_resmat1["helm_zs"]
    
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
        def _run_one(i):
            return cat_beta(sub_ys[i], sub_zs, device, item_budget)
        thetass = Parallel(n_jobs=max_workers)(delayed(_run_one)(i) for i in tqdm(range(sub_ys.shape[0])))
        thetass = torch.stack(thetass) # (n_models, budget)
        final_thetas = thetass[:, -1]
        
        # theta convergence
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(nrows=sub_ys.shape[0], ncols=1, figsize=(6, 2*sub_ys.shape[0]), sharex=True)
            budgets = np.arange(item_budget+1)
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
            # print(model)
            # if model == "OLMo-2-1124-7B-Instruct" or model == "Phi-4-mini-reasoning":
            #     continue
            model_data = sub_tensor[i]
            # plot of irt_corr_passat1s
            irt_probs = torch.sigmoid(float(final_thetas[i]) + sub_zs).cpu().numpy()
            # irt_probs = sub_zs.cpu().numpy()
            passat1s = np.nanmean(model_data, axis=-1)
            rho, pval = spearmanr(irt_probs, passat1s, nan_policy="omit")
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
            
            # law curve
            # mask = (irt_probs >= 0.02)
            # mask = (passat1s >= 0.01)
            # model_data = model_data[mask]
            # passat1s   = passat1s[mask]
            # irt_probs  = irt_probs[mask]

            pass_datk_gts = compute_pass_datk_gts(model_data)
            # pass_datk_gts = compute_pass_datk_irt(model_data, passat1s)
            pass_datk_irts = compute_pass_datk_irt(model_data, irt_probs)
            
            mae = np.mean(np.abs(pass_datk_gts - pass_datk_irts))
            n_samples = np.arange(1, model_data.shape[-1] + 1)
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                ax_left, ax_right = axes

                # Left: Pass@k vs k
                ax_left.plot(n_samples, pass_datk_gts, label="GT", linewidth=2)
                ax_left.plot(n_samples, pass_datk_irts, label="IRT Est", linewidth=2, linestyle="--")
                ax_left.set_xlabel("Number of Samples", fontsize=16)
                ax_left.set_ylabel("Pass@k", fontsize=16)
                ax_left.set_ylim(0, 1)
                ax_left.legend(fontsize=14)
                ax_left.tick_params(axis="both", labelsize=14)

                # Right: log–log of -log(Pass@k) vs k
                ax_right.loglog(n_samples, -np.log(pass_datk_gts),  label="GT", linewidth=2)
                ax_right.loglog(n_samples, -np.log(pass_datk_irts), label="IRT Est", linewidth=2, linestyle="--")
                ax_right.set_xlabel("Number of Samples", fontsize=16)
                ax_right.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
                ax_right.legend(fontsize=14)
                ax_right.tick_params(axis="both", labelsize=14)

                fig.suptitle(rf"{model}, {scen} (Pass@k MAE = {mae:.2f})", fontsize=16)
                fig.tight_layout()
                fig.savefig(f"{output_dir}/law_curve_{model}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)