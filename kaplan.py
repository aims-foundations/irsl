import pickle
import torch
torch.manual_seed(0)
from torch.distributions import Bernoulli
from tqdm import tqdm
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import argparse
import numpy as np
from scipy.optimize import curve_fit

def estimate_theta_all(asked_ys, asked_zs, device):
    def closure():
        optim.zero_grad()
        mask = ~torch.isnan(asked_ys)
        probs = torch.sigmoid(theta[:, None] + asked_zs[None, :])
        loss = -Bernoulli(probs=probs[mask]).log_prob(asked_ys[mask]).mean()
        loss.backward()
        return loss

    theta = torch.zeros((asked_ys.shape[0],), requires_grad=True, device=device)
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

# def power_law_func(flop, c, gamma, h):
#     return c * flop ** (-gamma) + h

def power_law_func(flop, w, b):
    return w * flop + b

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    # EleutherAI/pythia-6.9b, EleutherAI/pythia-12b
    # LLM360/Amber, allenai/OLMo-2-0325-32B, HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
    args = parser.parse_args()
    model_name = args.repo_id.split("/")[1]
    
    benchmark = "mmlu"
    scenario = "mmlu"

    device = "cuda:7"
    with open(f"data/gather_ckpt_data/results_{model_name}.pkl", "rb") as f:
        results = pickle.load(f)
    keep_cols = ~results.columns.get_level_values("z").isna()
    results = results.loc[:, keep_cols]
    results = results.loc[:, results.columns.get_level_values("benchmark") == benchmark]
    results = results.loc[:, results.columns.get_level_values("scenario") == scenario]
    results = results[~results.isna().all(axis=1)]
    ys = torch.tensor(results.values, dtype=torch.float, device=device)
    n_test_takers, n_items = ys.shape
    print(ys.shape)
    zs = results.columns.get_level_values("z").astype(float).to_numpy()
    zs = torch.tensor(zs, dtype=torch.float, device=device)
    time_steps = np.array([float(name.split("-")[-1]) for name in results.index])
    flops = time_steps * 2097152.0 * 1.7e9 * 6.0 / 1e21
    
    # sorted_indices = torch.argsort(zs, descending=True)
    # split_idx = n_items // 2
    # train_indices = sorted_indices[:split_idx]   # Smaller zs → train
    # test_indices = sorted_indices[split_idx:]    # Larger zs → test
    
    perm = torch.randperm(n_items)
    split_idx = n_items // 2
    train_indices = perm[:split_idx]
    test_indices = perm[split_idx:]
    
    # temperature = 9  # Tune this: lower = more extreme bias, higher = softer bias
    # split_idx = n_items // 2
    # eps = 1e-6
    # weights = ((zs.max() - zs) + eps) ** (1.0 / temperature)
    # probs = weights / weights.sum()
    # train_indices = torch.multinomial(probs, split_idx, replacement=False)
    # all_indices = torch.arange(n_items)
    # mask = torch.zeros(n_items, dtype=torch.bool)
    # mask[train_indices] = True
    # test_indices = all_indices[~mask]
        
    ys_train = ys[:, train_indices]
    ys_test = ys[:, test_indices]
    zs_train = zs[train_indices]
    zs_test = zs[test_indices]

    # gt
    gt_ctt_train = torch.nanmean(ys_train, dim=1).cpu().numpy()
    gt_ctt_test = torch.nanmean(ys_test, dim=1).cpu().numpy()
    
    # classic kaplan
    popt, _ = curve_fit(power_law_func, flops, gt_ctt_train)
    w, b = popt
    train_kaplans = power_law_func(flops, w, b)

    # irt
    train_theta = estimate_theta_all(ys_train, zs_train, device)
    train_probs = torch.sigmoid(train_theta[:, None] + zs_train[None, :])
    train_irts = torch.bernoulli(train_probs).mean(dim=1).cpu().numpy()
    test_probs = torch.sigmoid(train_theta[:, None] + zs_test[None, :])
    test_irts = torch.bernoulli(test_probs).mean(dim=1).cpu().numpy()
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # ————— Left subplot: training curves —————
        ax = axes[0]
        ax.plot(flops, gt_ctt_train,
                linestyle='-',
                color='black',
                linewidth=2,
                label='Ground truth')
        ax.plot(flops, train_kaplans,
                linestyle='--',
                label='Classic Kaplan')
        ax.plot(flops, train_irts,
                linestyle='--',
                label='1PL IRT')
        ax.set_xlabel("FLOPs (1e21)", fontsize=20)
        ax.set_ylabel("CTT", fontsize=20)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
        ax.set_title('Train', fontsize=16)

        # ————— Right subplot: test curves —————
        ax = axes[1]
        ax.plot(flops, gt_ctt_test,
                linestyle='-',
                color='black',
                linewidth=2,
                label='Ground truth')
        ax.plot(flops, train_kaplans,  # reuse LS fit
                linestyle='--',
                label='Classic Kaplan')
        ax.plot(flops, test_irts,
                linestyle='--',
                label='1PL IRT')
        ax.set_xlabel("FLOPs (1e21)", fontsize=20)
        ax.set_ylabel("CTT", fontsize=20)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
        ax.set_title('Test', fontsize=16)

        fig.tight_layout()
        fig.savefig(f"kaplan_{model_name}.png", dpi=300, bbox_inches="tight")
