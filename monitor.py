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

# def estimate_theta(asked_ys, asked_zs, device, theta_init=torch.zeros((1,))):
#     def closure():
#         optim.zero_grad()
#         mask = ~torch.isnan(asked_ys)
#         probs = torch.sigmoid(theta + asked_zs)
#         loss = -Bernoulli(probs=probs[mask]).log_prob(asked_ys[mask]).mean()
#         loss.backward()
#         return loss

#     theta = theta_init.clone().to(device).requires_grad_(True)
#     optim = torch.optim.LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
#     for iteration in range(100):
#         if iteration > 0:
#             previous_theta = theta.clone()
#             previous_loss = loss.clone()
        
#         loss = optim.step(closure)
        
#         if iteration > 0:
#             d_loss = previous_loss - loss
#             d_theta = torch.norm(previous_theta - theta, p=2)
#             grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
#             if d_loss < 1e-5 and d_theta < 1e-5 and grad_norm < 1e-5:
#                 break
    
#     return theta.detach()

# def compute_fisher_info(theta, remain_zs):
#     p = torch.sigmoid(theta + remain_zs)
#     return p * (1 - p)

# def adap_test(ys, zs, device, gt, budget=50):
#     adaptive_theta_hat = torch.zeros((1,), device=device)
#     adaptive_asked_zs = []
#     adaptive_asked_ys = []
#     remain_zs = zs.clone()
#     remain_ys = ys.clone()
    
#     pbar = tqdm(range(budget))
#     for _ in pbar:
#         fisher_info = compute_fisher_info(adaptive_theta_hat, remain_zs)
#         next_item = torch.argmax(fisher_info)
#         adaptive_asked_zs.append(remain_zs[next_item])
#         adaptive_asked_ys.append(remain_ys[next_item])
#         adaptive_theta_hat = estimate_theta_all(
#             torch.tensor(adaptive_asked_ys, device=device),
#             torch.tensor(adaptive_asked_zs, device=device),
#             device,
#             adaptive_theta_hat
#         )
#         pbar.set_postfix({
#             'adaptive': f"{adaptive_theta_hat.item():.2f}",
#             'gt': f"{gt.item():.2f}"
#         })
#         remain_zs = torch.cat([remain_zs[:next_item], remain_zs[next_item + 1:]])
#         remain_ys = torch.cat([remain_ys[:next_item], remain_ys[next_item + 1:]])
    
#     return adaptive_theta_hat

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

    train_step = int(n_test_takers * 0.1)
    gt_thetas = {}
    with open(f"AUC_{model_name}.txt", "w") as f:
        gt_theta = estimate_theta_all(ys, zs, device)
        theta_train = gt_theta[train_step]
        thetas_test = torch.tensor([theta_train] * (n_test_takers - train_step), device=device)
        ys_test = ys[train_step:, :]
        mask_test = ~torch.isnan(ys_test)
        probs_test = torch.sigmoid(thetas_test[:, None] + zs[None, :])
        auc_test = auroc(probs_test[mask_test], ys_test[mask_test])
        f.write(f"all, auc_test: {auc_test}\n")
        gt_thetas["all"] = gt_theta.cpu().numpy()
        
        auc_tests = []
        for scenario in tqdm(results.columns.get_level_values("scenario").unique()):
            mask = (results.columns.get_level_values("scenario") == scenario)
            ys_scenario = ys[:, mask]
            zs_scenario = zs[mask]
            gt_theta = estimate_theta_all(ys_scenario, zs_scenario, device)
            gt_thetas[ABBREVIATE[scenario]] = gt_theta.cpu().numpy()
            
            theta_train = gt_theta[train_step]
            thetas_test = torch.tensor([theta_train] * (n_test_takers - train_step), device=device)
            ys_test = ys_scenario[train_step:, :]
            mask_test = ~torch.isnan(ys_test)
            probs_test = torch.sigmoid(thetas_test[:, None] + zs_scenario[None, :])
            auc_test = auroc(probs_test[mask_test], ys_test[mask_test])
            auc_tests.append(auc_test)
            f.write(f"{scenario}, auc_test: {auc_test}\n")
        f.write(f"average, auc_test: {sum(auc_tests)/len(auc_tests)}")
        
    # adaptive_thetas = []
    # for i in tqdm(range(n_test_takers)):
        # gt theta
        # ys = ys[i, :]
        # gt_theta = estimate_theta(ys, zs, device)
        # gt_thetas.append(gt_theta.item())
        
        # adaptive theta
        # adaptive_theta = adap_test(ys, zs, device, gt_theta)
        # adaptive_thetas.append(adaptive_theta.item())
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(12, 8))

        # Ensure time_steps contains a mix of int and str, sort only the numeric parts
        x = list(range(len(time_steps)))  # x-axis positions
        x_labels = time_steps  # corresponding labels
        
        for abbrev, thetas in gt_thetas.items():
            plt.plot(x, thetas, marker="o", label=abbrev)

        plt.xticks(x, x_labels, rotation=45)
        plt.xlabel("Time Step", fontsize=25)
        plt.ylabel("Estimated Theta", fontsize=25)
        # plt.tick_params(axis="both", labelsize=16)
        plt.legend(title="Scenario", loc="best")
        plt.savefig(f"result/monitor_{model_name}.png", dpi=300, bbox_inches="tight")
