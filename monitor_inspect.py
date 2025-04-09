import pickle
import torch
torch.manual_seed(0)
from torch.distributions import Bernoulli
from tqdm import tqdm
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import argparse
from torchmetrics import AUROC
auroc = AUROC(task="binary")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    # EleutherAI/pythia-6.9b, EleutherAI/pythia-12b
    # LLM360/Amber, allenai/OLMo-2-0325-32B, HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
    args = parser.parse_args()
    model_name = args.repo_id.split("/")[1]
    
    device = "cuda:7"
    with open(f"data/gather_ckpt_data/results_{model_name}.pkl", "rb") as f:
        results = pickle.load(f)
    keep_cols = ~results.columns.get_level_values("z").isna()
    results = results.loc[:, keep_cols]
    ys = torch.tensor(results.values, dtype=torch.float, device=device)
    n_test_takers, n_items = ys.shape
    zs = results.columns.get_level_values("z").astype(float).to_numpy()
    zs = torch.tensor(zs, dtype=torch.float, device=device)
    time_steps = [name.split("-")[-1] for name in results.index]

    gt_thetas = {}
    gt_ctts = {}
    with open(f"AUC_{model_name}.txt", "w") as f:
        gt_theta = estimate_theta_all(ys, zs, device)
        gt_thetas["all"] = gt_theta.cpu().numpy()
        
        for scenario in tqdm(results.columns.get_level_values("scenario").unique()):
            if scenario == "mmlu":
                mask = (results.columns.get_level_values("scenario") == scenario)
                ys_scenario = ys[:, mask]
                zs_scenario = zs[mask]
                gt_theta = estimate_theta_all(ys_scenario, zs_scenario, device)
                gt_thetas[scenario] = torch.sigmoid(gt_theta).cpu().numpy()
                
                gt_ctt = torch.nanmean(ys_scenario, dim=1)
                gt_ctts[scenario] = gt_ctt.cpu().numpy()
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(12, 8))

        # Ensure time_steps contains a mix of int and str, sort only the numeric parts
        x = list(range(len(time_steps)))  # x-axis positions
        x_labels = time_steps  # corresponding labels
        
        plt.plot(x, gt_thetas["mmlu"], marker="o", label="IRT ability")
        plt.plot(x, gt_ctts["mmlu"], marker="x", label="CTT")

        plt.xticks(x, x_labels, rotation=45)
        plt.xlabel("Time Step", fontsize=25)
        # plt.ylabel("Estimated Theta", fontsize=25)
        # plt.tick_params(axis="both", labelsize=16)
        plt.legend()
        plt.savefig(f"result/monitor_inspect_{model_name}.png", dpi=300, bbox_inches="tight")
