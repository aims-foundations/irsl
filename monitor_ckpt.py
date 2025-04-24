import os
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
import concurrent.futures

ABBREVIATE = {
    "wikifact": "wiki",
    "synthetic_reasoning": "srea",
    "truthful_qa": "truth",
    "math": "math",
    "gsm": "gsm",
    "babi_qa": "babi",
    "bbq": "bbq",
    "thai_exam": "thai",
    "legal_support": "lsup",
    "legalbench": "lben",
    "civil_comments": "civ",
    "dyck_language_np=3": "dyck",
    "air_bench_2024": "air",
    "med_qa": "med",
    "raft": "rft",
    "mmlu": "mmlu",
    "entity_matching": "emat",
    "boolq": "bool",
    "entity_data_imputation": "eimp",
    "commonsense": "comm",
    "imdb": "imdb",
    "blimp": "blimp"
}

def estimate_theta(asked_ys, asked_zs, device, theta_init=torch.zeros((1,))):
    def closure():
        optim.zero_grad()
        mask = ~torch.isnan(asked_ys_)
        probs = torch.sigmoid(theta[:, None] + asked_zs[None, :])
        loss = -Bernoulli(probs=probs[mask]).log_prob(asked_ys_[mask]).mean()
        loss.backward()
        return loss

    asked_ys_ = asked_ys if asked_ys.ndim > 1 else asked_ys.unsqueeze(0)
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
    
    return theta.detach().clone()

def elo_update(asked_ys, asked_zs, theta_init, k=0.2):
    if torch.isnan(asked_ys[-1:]):
        return theta_init
    p = torch.sigmoid(theta_init + asked_zs[-1:])
    return theta_init + k * (asked_ys[-1:] - p)

def adap_test(ys, zs, device, burn_in, budget, adaptive_theta_hat, verbose=True, elo=False):
    adaptive_theta_hat = adaptive_theta_hat.to(device)
    remain_zs = zs.clone()
    remain_ys = ys.clone()
    
    perm = torch.randperm(len(remain_zs))[:burn_in]
    adaptive_asked_zs = remain_zs[perm].tolist()
    adaptive_asked_ys = remain_ys[perm].tolist()
    mask = torch.ones(len(zs), dtype=torch.bool, device=device)
    mask[perm] = False
    remain_zs = remain_zs[mask]
    remain_ys = remain_ys[mask]

    adaptive_theta_hat = estimate_theta(
        torch.tensor(adaptive_asked_ys, device=device),
        torch.tensor(adaptive_asked_zs, device=device),
        device,
        adaptive_theta_hat,
    )
    
    # adaptive_asked_zs = []
    # adaptive_asked_ys = []
    
    adaptive_theta_hats = []
    update_fn = elo_update if elo else lambda y, z, t: estimate_theta(y, z, device, t)
    pbar = tqdm(range(budget)) if verbose else range(budget)
    for _ in pbar:
        next_index = torch.argmin(torch.abs(remain_zs + adaptive_theta_hat)).item()
        adaptive_asked_zs.append(remain_zs[next_index])
        adaptive_asked_ys.append(remain_ys[next_index])
        # print(adaptive_theta_hat.item(), remain_zs[next_index].item(), remain_ys[next_index].item())
        adaptive_theta_hat = update_fn(
            torch.tensor(adaptive_asked_ys, device=device),
            torch.tensor(adaptive_asked_zs, device=device),
            adaptive_theta_hat,
        )
        adaptive_theta_hats.append(adaptive_theta_hat)
        remain_zs = torch.cat([remain_zs[:next_index], remain_zs[next_index + 1:]])
        remain_ys = torch.cat([remain_ys[:next_index], remain_ys[next_index + 1:]])
    
    if elo:
        final_theta = estimate_theta(
            torch.tensor(adaptive_asked_ys, device=device),
            torch.tensor(adaptive_asked_zs, device=device),
            device,
            adaptive_theta_hat
        )
        adaptive_theta_hats[-1] = final_theta
    
    # print(adaptive_theta_hats)
    return adaptive_theta_hats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="/pythia-6.9b_legalbench")
    # EleutherAI/pythia-6.9b, EleutherAI/pythia-12b
    # LLM360/Amber, allenai/OLMo-2-0325-32B, HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
    parser.add_argument("--gpuid", type=int, default=3)
    parser.add_argument("--max_workers", type=int, default=256)
    parser.add_argument("--burn_in", type=int, default=10)
    parser.add_argument("--budget", type=int, default=50)
    args = parser.parse_args()
    model_name = args.repo_id.split("/")[1]
    
    output_dir = f"result/monitor/{model_name}_budget_{args.budget}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = f"cuda:{args.gpuid}"
    with open(f"data/gather_ckpt_data/results_{model_name}.pkl", "rb") as f:
        results = pickle.load(f)
    keep_cols = ~results.columns.get_level_values("z").isna()
    results = results.loc[:, keep_cols]
    ys = torch.tensor(results.values, dtype=torch.float, device=device)
    n_test_takers, n_items = ys.shape
    print(ys.shape)
    zs = results.columns.get_level_values("z").astype(float).to_numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(zs, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title("Distribution of Selected z")
    plt.xlabel("z")
    plt.savefig(f"z_distri.png", dpi=300, bbox_inches='tight')
    zs = torch.tensor(zs, dtype=torch.float, device=device)
    
    # sampled_indices = torch.randperm(n_items)[:1000]
    # ys = ys[12:17][:, sampled_indices]
    # ys = ys[:20]
    # zs = zs[sampled_indices]
    n_test_takers, n_items = ys.shape
    print(ys.shape)

    # gt_theta
    gt_thetas = estimate_theta(ys, zs, device, theta_init=torch.zeros((n_test_takers,))) # shape: (n_test_takers,)
    torch.save(gt_thetas.cpu(), f"{output_dir}/gt_thetas.pt")

    # random
    perm_indices = torch.randperm(n_items)
    permuted_ys = ys[:, perm_indices]
    permuted_zs = zs[perm_indices]
    def estimate_theta_step(i):
        return estimate_theta(
            permuted_ys[:, :i+args.burn_in],
            permuted_zs[:i+args.burn_in],
            # permuted_ys[:, :i],
            # permuted_zs[:i],
            device,
            theta_init=torch.zeros((n_test_takers,))
        )
    random_thetass = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(estimate_theta_step, i) for i in range(1, args.budget + 1)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            random_thetass.append(future.result())
    random_thetass = torch.stack(random_thetass, dim=1) # shape: (n_test_takers, budget)
    torch.save(random_thetass.cpu(), f"{output_dir}/random_thetas.pt")
    
    # # adaptive
    # adaptive_thetass = []
    # adaptive_theta_hat = torch.zeros((1,))
    # for i in tqdm(range(n_test_takers)):
    #     single_ys = ys[i, :]
    #     adaptive_thetas = adap_test(single_ys, zs, device, args.burn_in, args.budget, adaptive_theta_hat)
    #     adaptive_theta_hat = adaptive_thetas[-1].clone()
    #     adaptive_thetas = torch.cat(adaptive_thetas)
    #     adaptive_thetass.append(adaptive_thetas)
    # adaptive_thetass = torch.stack(adaptive_thetass, dim=0) # shape: (n_test_takers, budget)
    # torch.save(adaptive_thetass.cpu(), f"{output_dir}/adaptive_thetas.pt")
    
    # elo
    elo_thetass = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i in range(n_test_takers):
            futures.append(executor.submit(
                lambda idx, ys_i: torch.cat(adap_test(ys_i, zs, device, args.burn_in, args.budget, torch.zeros((1,)), verbose=False, elo=True)),
                i, ys[i, :]
            ))
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            elo_thetass.append(f.result())
    elo_thetass = torch.stack(elo_thetass, dim=0)
    torch.save(elo_thetass.cpu(), f"{output_dir}/elo_thetas.pt")

    # elo_thetass = []
    # elo_theta_hat = torch.zeros((1,))
    # for i in tqdm(range(n_test_takers)):
    #     single_ys = ys[i, :]
    #     elo_thetas = adap_test(single_ys, zs, device, args.burn_in, args.budget, elo_theta_hat, verbose=False, elo=True)
    #     elo_theta_hat = elo_thetas[-1].clone()
    #     elo_thetas = torch.cat(elo_thetas)
    #     elo_thetass.append(elo_thetas)
    #     # print(elo_thetas)
    # elo_thetass = torch.stack(elo_thetass, dim=0) # shape: (n_test_takers, budget)
    # torch.save(elo_thetass.cpu(), f"{output_dir}/elo_thetas.pt")