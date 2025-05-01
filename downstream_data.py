import pickle
import torch
torch.manual_seed(0)
from torch.distributions import Bernoulli
from tqdm import tqdm
from collections import defaultdict
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

def linear_func(flop, w, b):
    return w * flop + b

if __name__ == "__main__":
    device = "cuda:1"
    repo_ids = [
        "EleutherAI/pythia-12b",
        "EleutherAI/pythia-6.9b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-14m",
        "LLM360/Amber",
        "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints",
        "HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints",
        "HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints",
    ]
    model_names = [repo_id.split("/")[1] for repo_id in repo_ids]
    
    scenarios = ['babi_qa', 'civil_comments', 'commonsense',
       'dyck_language_np=3', 'entity_data_imputation', 'entity_matching',
       'gsm', 'legal_support', 'legalbench', 'mmlu', 'raft',
       'synthetic_reasoning', 'wikifact'] # 'med_qa', 'boolq', 'imdb'
    
    results_dict = defaultdict(lambda: defaultdict(dict))
    for scenario in tqdm(scenarios):
        for model_name in model_names:
            with open(f"data/gather_ckpt_data/aggregate_matrix/results_{model_name}.pkl", "rb") as f:
                results = pickle.load(f)
            keep_cols = ~results.columns.get_level_values("z").isna()
            results = results.loc[:, keep_cols]
            # results = results.loc[:, results.columns.get_level_values("benchmark") == benchmark]
            results = results.loc[:, results.columns.get_level_values("scenario") == scenario]
            results = results[~results.isna().all(axis=1)]
            ys = torch.tensor(results.values, dtype=torch.float, device=device)
            n_test_takers, n_items = ys.shape
            zs = results.columns.get_level_values("z").astype(float).to_numpy()
            zs = torch.tensor(zs, dtype=torch.float, device=device)
            time_steps = np.array([float(name.split("-")[-1]) for name in results.index])
            # flops = time_steps * 2097152.0 * 1.7e9 * 6.0 / 1e21

            perm = torch.randperm(n_items)
            split_idx = n_items // 2
            train_indices = perm[:split_idx]
            test_indices = perm[split_idx:]
        
            ys_train = ys[:, train_indices]
            ys_test = ys[:, test_indices]
            zs_train = zs[train_indices]
            zs_test = zs[test_indices]

            # gt
            gt_ctt_train = torch.nanmean(ys_train, dim=1).cpu().numpy()
            gt_ctt_test = torch.nanmean(ys_test, dim=1).cpu().numpy()
            
            # classic linear
            popt, _ = curve_fit(linear_func, time_steps, gt_ctt_train)
            w, b = popt
            train_linears = linear_func(time_steps, w, b)

            # irt
            train_theta = estimate_theta_all(ys_train, zs_train, device)
            train_probs = torch.sigmoid(train_theta[:, None] + zs_train[None, :])
            # train_irts = torch.bernoulli(train_probs).mean(dim=1).cpu().numpy()
            train_irts = train_probs.mean(dim=1).cpu().numpy()
            test_probs = torch.sigmoid(train_theta[:, None] + zs_test[None, :])
            # test_irts = torch.bernoulli(test_probs).mean(dim=1).cpu().numpy()
            test_irts = test_probs.mean(dim=1).cpu().numpy()
            
            results_dict[scenario][model_name] = {
                "gt_ctt_train": gt_ctt_train,
                "gt_ctt_test": gt_ctt_test,
                "train_linears": train_linears,
                "train_irts": train_irts,
                "test_irts": test_irts,
            }
            
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"downstream_data.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)