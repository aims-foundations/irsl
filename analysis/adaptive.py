import pickle
import torch
torch.manual_seed(0)
from torch.distributions import Bernoulli
from tqdm import tqdm
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import argparse

def estimate_theta(asked_ys, asked_zs, device, theta_init=torch.zeros((1,))):
    def closure():
        optim.zero_grad()
        mask = ~torch.isnan(asked_ys)
        probs = torch.sigmoid(theta + asked_zs)
        loss = -Bernoulli(probs=probs[mask]).log_prob(asked_ys[mask]).mean()
        loss.backward()
        return loss

    theta = theta_init.clone().to(device).requires_grad_(True)
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

def compute_fisher_info(theta, remain_zs):
    p = torch.sigmoid(theta + remain_zs)
    return p * (1 - p)

def adap_test(ys, zs, device, gt, budget=50):
    adaptive_theta_hat = torch.zeros((1,), device=device)
    adaptive_asked_zs = []
    adaptive_asked_ys = []
    remain_zs = zs.clone()
    remain_ys = ys.clone()
    
    pbar = tqdm(range(budget))
    for _ in pbar:
        fisher_info = compute_fisher_info(adaptive_theta_hat, remain_zs)
        next_item = torch.argmax(fisher_info)
        adaptive_asked_zs.append(remain_zs[next_item])
        adaptive_asked_ys.append(remain_ys[next_item])
        adaptive_theta_hat = estimate_theta(
            torch.tensor(adaptive_asked_ys, device=device),
            torch.tensor(adaptive_asked_zs, device=device),
            device,
            adaptive_theta_hat
        )
        pbar.set_postfix({
            'adaptive': f"{adaptive_theta_hat.item():.2f}",
            'gt': f"{gt.item():.2f}"
        })
        remain_zs = torch.cat([remain_zs[:next_item], remain_zs[next_item + 1:]])
        remain_ys = torch.cat([remain_ys[:next_item], remain_ys[next_item + 1:]])
    
    return adaptive_theta_hat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    # EleutherAI/pythia-6.9b, EleutherAI/pythia-12b, LLM360/Amber
    args = parser.parse_args()
    
    model2pattern = {
        "EleutherAI/pythia-6.9b": "step",
        "EleutherAI/pythia-12b": "step",
        "LLM360/Amber": "ckpt_"
    }
    pattern = model2pattern[args.repo_id]
    
    device = "cuda:4"
    with open(f"../data/results_{args.repo_id.split('/')[1]}.pkl", "rb") as f:
        results = pickle.load(f)
    keep_cols = ~results.columns.get_level_values("z").isna()
    results = results.loc[:, keep_cols]

    data = torch.tensor(results.values, dtype=torch.float, device=device)
    n_test_takers, n_items = data.shape
    zs = results.columns.get_level_values("z").astype(float).to_numpy()
    zs = torch.tensor(zs, dtype=torch.float, device=device)
    time_steps = [int(name.split(pattern)[-1]) for name in results.index]
    
    gt_thetas = []
    adaptive_thetas = []
    for i in tqdm(range(n_test_takers)):
        # gt theta
        ys = data[i, :]
        gt_theta = estimate_theta(ys, zs, device)
        gt_thetas.append(gt_theta.item())
        
        # adaptive theta
        # adaptive_theta = adap_test(ys, zs, device, gt_theta)
        # adaptive_thetas.append(adaptive_theta.item())
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(8, 6))
        plt.plot(time_steps, gt_thetas, marker="o", linestyle="-", label="Ground Truth Theta")
        # plt.plot(time_steps, adaptive_thetas, marker="x", linestyle="--", label="Adaptive Theta")
        plt.xlabel("Time Step")
        plt.ylabel("Estimated Theta")
        plt.savefig("../result/adaptive.png", dpi=300)
