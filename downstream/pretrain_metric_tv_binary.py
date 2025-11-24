import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error                
from scipy.stats import spearmanr
from tueplots import bundles
bundles.icml2024()

def backtest_krr(x, y, A):
    x = x.reshape(-1, 1)
    n = len(y)
    split = int(A * n)
    x_tr, y_tr_gt = x[:split], y[:split]
    x_te, y_te_gt = x[split:], y[split:]
    kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
    kr.fit(x_tr, y_tr_gt)
    y_tr_pred = kr.predict(x_tr)
    y_te_pred = kr.predict(x_te)
    train_mae = mean_absolute_error(y_tr_pred, y_tr_gt) / (y.max() - y.min())
    test_mae = mean_absolute_error(y_te_pred, y_te_gt) / (y.max() - y.min())
    return (train_mae, test_mae)

def total_variance(a):
    n = len(a)
    num = np.abs(np.diff(a)).sum()
    denom = abs(a[-1] - a[0])
    return (n / (n - 1.0)) * (num / denom)

if __name__ == "__main__":
    RESULT_PATH = "/lfs/skampere2/0/sttruong/irsl/result/pretrain_binarycat_helm_full_backup/result.pkl"
    DROP_FIRST_FRAC = 0.10  # drop the first 10% of checkpoints (same idea as your reference)
    BUDGET = 50
    
    with open(RESULT_PATH, "rb") as f:
        results_dict = pickle.load(f)
        
    acc_tvs   = []
    theta_tvs = []
    rows = []
    rng = np.random.default_rng(0)  # reproducible subset

    for scenario, model_dict in results_dict.items():
        for model, payload in model_dict.items():
            ys      = torch.tensor(payload["resmat"].values)               # torch.Tensor (n_checkpoints, n_items)
            thetass = payload["thetass"]           # torch.Tensor (n_checkpoints, budget)

            final_thetas = thetass[:, -1].cpu().numpy()
            means_sub = torch.nanmean(ys[:, torch.randperm(ys.shape[1])[:BUDGET]], dim=1).cpu().numpy()

            # optional warm-up drop, like in your reference
            drop = int(DROP_FIRST_FRAC * len(final_thetas))
            final_thetas = final_thetas[drop:]
            means_sub    = means_sub[drop:]

            acc_tv   = total_variance(means_sub)
            theta_tv = total_variance(final_thetas)

            acc_tvs.append(acc_tv)
            theta_tvs.append(theta_tv)

    # Averages across all (scenario, model) pairs
    acc_tv_avg   = float(np.mean(acc_tvs))
    theta_tv_avg = float(np.mean(theta_tvs))

    print(f"=== TV averages across all datasets & models ===")
    print(f"Accuracy TV avg: {acc_tv_avg:.4f}")
    print(f"Theta TV avg:    {theta_tv_avg:.4f}\n")

        
    
    # steps = results_dict[scenario][model]["steps"]
    # ys = results_dict[scenario][model]["ys"]
    # thetass = results_dict[scenario][model]["thetass"]
    
    # step_pcts = steps.astype(int) / steps.astype(int).max() * 100.0
    # final_thetas = thetass[:, -1].cpu().numpy()
    # means_sub = torch.nanmean(ys[:, torch.randperm(ys.shape[1])[:budget]], dim=1).cpu().numpy()

    # # drop first DROP_FIRST_FRAC%
    # drop_idx = int(DROP_FIRST_FRAC * len(step_pcts))
    # step_pcts, final_thetas, means_sub = step_pcts[drop_idx:], final_thetas[drop_idx:], means_sub[drop_idx:]

    # # back testing
    # acc_maes  = [backtest_krr(step_pcts, means_sub, A) for A in A_LIST]
    # theta_maes = [backtest_krr(step_pcts, final_thetas, A) for A in A_LIST]
    # # print results
    # print("=== Backtesting Results ===")
    # for A, (train_acc, test_acc), (train_theta, test_theta) in zip(A_LIST, acc_maes, theta_maes):
    #     print(f"A={A:.1f}\n    Accuracy: train MAE={train_acc:.4f}, test MAE={test_acc:.4f}\n"
    #         f"    Theta: train MAE={train_theta:.4f}, test MAE={test_theta:.4f}")

    # # variance
    # acc_tv   = total_variance(means_sub)
    # theta_tv = total_variance(final_thetas)
    # print("\n=== Variance Results ===")
    # print(f"Accuracy TV: {acc_tv:.4f}, Theta TV: {theta_tv:.4f}")