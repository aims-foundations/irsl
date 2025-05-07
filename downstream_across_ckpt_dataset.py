import pickle
import torch
torch.manual_seed(0)
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

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

if __name__ == "__main__":
    device = "cuda:0"
    repo_id = "EleutherAI/pythia-12b"
    # EleutherAI/pythia-6.9b, EleutherAI/pythia-12b
    # LLM360/Amber, HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
    model_name = repo_id.split("/")[1]
    
    with open(f"data/gather_ckpt_data/aggregate_matrix/results_{model_name}.pkl", "rb") as f:
        results = pickle.load(f)
    keep_cols = ~results.columns.get_level_values("z").isna()
    results = results.loc[:, keep_cols]
    time_steps = np.array([float(name.split("-")[-1]) for name in results.index])
    # flops = time_steps * 2097152.0 * 1.7e9 * 6.0 / 1e21
    
    # scenarios = ['babi_qa', 'civil_comments', 'commonsense',
    #     'dyck_language_np=3', 'entity_data_imputation', 'entity_matching',
    #     'gsm', 'legal_support', 'legalbench', 'mmlu', 'raft',
    #     'synthetic_reasoning', 'wikifact'] # 'med_qa', 'boolq', 'imdb'
    # plt.figure(figsize=(12, 6))
    # for scenario in tqdm(scenarios):
    #     sub_results = results.loc[:, results.columns.get_level_values("scenario") == scenario]
    #     sub_results = sub_results[~sub_results.isna().all(axis=1)]
    #     ys = torch.tensor(sub_results.values, dtype=torch.float, device=device)
    #     ctt = torch.nanmean(ys, dim=1).cpu().numpy()
    #     print(ctt.shape)
    #     plt.plot(time_steps, ctt, label=scenario)
    # plt.xlabel("Time Step")
    # plt.ylabel("Mean Accuracy")
    # plt.legend(loc="best", fontsize=9)
    # plt.tight_layout()
    # plt.savefig(f"ctts_{model_name}.png", dpi=300, bbox_inches="tight")

    scenario1 = "legalbench"
    scenario2 = "legal_support"
    
    results1 = results.loc[:, results.columns.get_level_values("scenario") == scenario1]
    results1 = results1[~results1.isna().all(axis=1)]
    ys1 = torch.tensor(results1.values, dtype=torch.float, device=device)
    n_test_takers, n_items1 = ys1.shape
    print(ys1.shape)
    zs1 = results1.columns.get_level_values("z").astype(float).to_numpy()
    zs1 = torch.tensor(zs1, dtype=torch.float, device=device)
    split_idx = n_test_takers // 2
    ys1_train, ys1_test = ys1[:split_idx, :], ys1[split_idx:, :]
    
    results2 = results.loc[:, results.columns.get_level_values("scenario") == scenario2]
    results2 = results2[~results2.isna().all(axis=1)]
    ys2 = torch.tensor(results2.values, dtype=torch.float, device=device)
    n_test_takers2, n_items2 = ys2.shape
    print(ys2.shape)
    assert n_test_takers2 == n_test_takers
    zs2 = results2.columns.get_level_values("z").astype(float).to_numpy()
    zs2 = torch.tensor(zs2, dtype=torch.float, device=device)
    ys2_train, ys2_test = ys2[:split_idx, :], ys2[split_idx:, :]
    
    # gt
    gt_ctt1 = torch.nanmean(ys1, dim=1).cpu().numpy()
    gt_ctt2 = torch.nanmean(ys2, dim=1).cpu().numpy()
    
    # irt
    train_thetas = estimate_theta_all(ys1_train, zs1, device)
    train_timesteps, test_timesteps = time_steps[:split_idx], time_steps[split_idx:]
    n_train, n_test = len(train_timesteps), len(test_timesteps)
    # ## AR
    # ar = AutoReg(train_thetas.cpu().numpy(), lags=10, old_names=False).fit()
    # pred = ar.predict(start=n_train, end=n_train+n_test-1, dynamic=False)
    ## ARIMA
    arima = ARIMA(train_thetas.cpu().numpy(), order=(2,1,2)).fit() # (p,d,q)
    pred = arima.forecast(steps=n_test)
    test_thetas = torch.tensor(pred, dtype=torch.float, device=device)
    
    probs1_train = torch.sigmoid(train_thetas[:, None] + zs1[None, :])
    irts1_train = probs1_train.mean(dim=1).cpu().numpy()
    probs1_test = torch.sigmoid(test_thetas[:, None] + zs1[None, :])
    irts1_test = probs1_test.mean(dim=1).cpu().numpy()
    probs2_train = torch.sigmoid(train_thetas[:, None] + zs2[None, :])
    irts2_train = probs2_train.mean(dim=1).cpu().numpy()
    probs2_test = torch.sigmoid(test_thetas[:, None] + zs2[None, :])
    irts2_test = probs2_test.mean(dim=1).cpu().numpy()
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        ax = axes[0]
        ax.plot(time_steps, gt_ctt1,
                linestyle='-',
                color='black',
                linewidth=2,
                label='Ground truth')
        ax.plot(train_timesteps, irts1_train,
                linestyle='--',
                label=f"Train")
        ax.plot(test_timesteps, irts1_test,
                linestyle='--',
                label=f"Generalize Checkpoint")
        ax.set_xlabel("Time Step", fontsize=20)
        ax.set_ylabel("Mean Accuracy", fontsize=20)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
        ax.set_title(scenario1, fontsize=16)

        ax = axes[1]
        ax.plot(time_steps, gt_ctt2,
                linestyle='-',
                color='black',
                linewidth=2,
                label='Ground truth')
        ax.plot(train_timesteps, irts2_train,
                linestyle='--',
                label=f"Generalize Dataset")
        ax.plot(test_timesteps, irts2_test,
                linestyle='--',
                label=f"Double Generalize")
        ax.set_xlabel("Time Step", fontsize=20)
        ax.set_ylabel("Mean Accuracy", fontsize=20)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
        ax.set_title(scenario2, fontsize=16)

        fig.tight_layout()
        fig.savefig(f"downstream_across_ckpt_dataset_{model_name}_{scenario1}_{scenario2}.png", dpi=300, bbox_inches="tight")
