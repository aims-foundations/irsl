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
from sklearn.metrics import mean_squared_error

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

def linear_func(flop, w, b):
    return w * flop + b

# def sigmoid_func(flop, a, b):
#     return 1.0 / (1.0 + np.exp(-a * (flop - b)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    # EleutherAI/pythia-6.9b, EleutherAI/pythia-12b
    # LLM360/Amber, allenai/OLMo-2-0325-32B, HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
    args = parser.parse_args()
    model_name = args.repo_id.split("/")[1]
    
    # ['commonsense', 'gsm', 'legalbench', 'med_qa', 'mmlu']
    benchmark = "lite"
    scenario1 = "legalbench"
    scenario2 = "mmlu"

    device = "cuda:7"
    with open(f"data/gather_ckpt_data/results_{model_name}.pkl", "rb") as f:
        results = pickle.load(f)
    keep_cols = ~results.columns.get_level_values("z").isna()
    results = results.loc[:, keep_cols]
    results = results.loc[:, results.columns.get_level_values("benchmark") == benchmark]
    time_steps = np.array([float(name.split("-")[-1]) for name in results.index])
    flops = time_steps * 2097152.0 * 1.7e9 * 6.0 / 1e21
    
    results1 = results.loc[:, results.columns.get_level_values("scenario") == scenario1]
    results1 = results1[~results1.isna().all(axis=1)]
    ys_train = torch.tensor(results1.values, dtype=torch.float, device=device)
    n_test_takers1, n_items1 = ys_train.shape
    print(ys_train.shape)
    zs_train = results1.columns.get_level_values("z").astype(float).to_numpy()
    zs_train = torch.tensor(zs_train, dtype=torch.float, device=device)
    
    results2 = results.loc[:, results.columns.get_level_values("scenario") == scenario2]
    results2 = results2[~results2.isna().all(axis=1)]
    ys_test = torch.tensor(results2.values, dtype=torch.float, device=device)
    n_test_takers2, n_items2 = ys_test.shape
    print(ys_test.shape)
    zs_test = results2.columns.get_level_values("z").astype(float).to_numpy()
    zs_test = torch.tensor(zs_test, dtype=torch.float, device=device)

    # gt
    gt_ctt_train = torch.nanmean(ys_train, dim=1).cpu().numpy()
    gt_ctt_test = torch.nanmean(ys_test, dim=1).cpu().numpy()
    
    # classic kaplan
    popt, _ = curve_fit(linear_func, flops, gt_ctt_train)
    w, b = popt
    train_kaplans = linear_func(flops, w, b)

    # irt
    train_theta = estimate_theta_all(ys_train, zs_train, device)
    train_probs = torch.sigmoid(train_theta[:, None] + zs_train[None, :])
    # train_irts = torch.bernoulli(train_probs).mean(dim=1).cpu().numpy()
    train_irts = train_probs.mean(dim=1).cpu().numpy()
    test_probs = torch.sigmoid(train_theta[:, None] + zs_test[None, :])
    # test_irts = torch.bernoulli(test_probs).mean(dim=1).cpu().numpy()
    test_irts = test_probs.mean(dim=1).cpu().numpy()
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        mse_linear_train = mean_squared_error(gt_ctt_train, train_kaplans)
        mse_irt_train    = mean_squared_error(gt_ctt_train, train_irts)
        mse_linear_test  = mean_squared_error(gt_ctt_test, train_kaplans)
        mse_irt_test     = mean_squared_error(gt_ctt_test, test_irts)
    
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
                label='Linear')
        ax.plot(flops, train_irts,
                linestyle='--',
                label='1PL IRT')
        ax.set_xlabel("FLOPs (1e21)", fontsize=20)
        ax.set_ylabel("CTT", fontsize=20)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
        axes[0].set_title(
            "Train: MSE_{{linear}}={:.3e}, MSE_{{IRT}}={:.3e}".format(
                mse_linear_train, mse_irt_train
            ),
            fontsize=16,
        )

        # ————— Right subplot: test curves —————
        ax = axes[1]
        ax.plot(flops, gt_ctt_test,
                linestyle='-',
                color='black',
                linewidth=2,
                label='Ground truth')
        ax.plot(flops, train_kaplans,  # reuse LS fit
                linestyle='--',
                label='Linear')
        ax.plot(flops, test_irts,
                linestyle='--',
                label='1PL IRT')
        ax.set_xlabel("FLOPs (1e21)", fontsize=20)
        ax.set_ylabel("CTT", fontsize=20)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
        axes[1].set_title(
            "Test:  MSE_{{linear}}={:.3e}, MSE_{{IRT}}={:.3e}".format(
                mse_linear_test, mse_irt_test
            ),
            fontsize=16,
        )

        fig.tight_layout()
        fig.savefig(f"kaplan_across_dataset_{scenario1}_{scenario2}_{model_name}.png", dpi=300, bbox_inches="tight")
