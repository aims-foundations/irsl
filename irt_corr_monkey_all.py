import pandas as pd
from tqdm import tqdm
import torch
from torch.distributions import Bernoulli
from torch.optim import LBFGS
import pickle
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
from scipy.optimize import curve_fit
import os
from scipy.stats import pearsonr

probs_holder = {'current': None}

def trainer(parameters, optim, closure, n_iter=100, verbose=True):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for iteration in pbar:
        if iteration > 0:
            # Clone each tensor individually for previous state
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
            previous_probs = current_probs
        
        loss = optim.step(closure)
        current_probs = probs_holder['current']
        
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            d_probs = torch.norm(previous_probs - current_probs, p=2).item()
            if verbose:
                pbar.set_postfix({"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss, "d_probs": d_probs})
            
            if d_loss < 1e-4 and d_parameters < 1e-5 and grad_norm < 1e-5 and d_probs < 1e-5:
                break
            
    return parameters

def estimate_success_rate_at_k_per_problem(n: int, c: int, k: int) -> float:
    """
    :param n: number of total attempts on this problem.
    :param c: number of correct attempts on this problem.
    :param k: k in pass_i@$k$.
    """
    if n - c < k: 
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
                              
def power_law_func(k, a, b):
    return a * k ** (-b)

if __name__ == "__main__":
    monkey_model_name = "Llama-3-8B-Instruct" # "Llama-3-70B-Instruct"# "Pythia-6.9B"
    monkey_scenario = "GSM8K" # "MATH"
    helm_filename = "lite_gsm_results" # "classic_math_results" # "classic_gsm_results" 
    helm_model_name = "meta/llama-3-8b" # "eleutherai/pythia-6.9b" "meta/llama-3-70b"
    
    monkey_dataset = pd.read_json(f"hf://datasets/ScalingIntelligence/monkey_business/{monkey_scenario}_{monkey_model_name}.json")
    monkey_questions2iscorrects = {
        row["question"]: row["is_corrects"]
        for _, row in monkey_dataset.iterrows()
    }
    
    with open(f"/lfs/skampere1/0/sttruong/deval/gather_helm_data/{helm_filename}.pkl", "rb") as f:
        helm_resmat = pickle.load(f)
    helm_questions = list(helm_resmat.columns.get_level_values("input.text"))
    zs = helm_resmat.columns.get_level_values("z").astype(float)
    plt.figure(figsize=(8, 4))
    plt.hist(zs, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title("Distribution of Selected z")
    plt.xlabel("z")
    plt.savefig(f"z_distri_{monkey_model_name}_{monkey_scenario}.png", dpi=300, bbox_inches='tight')
    
    helm_question_set = set(helm_questions)
    monkey_question_set = set(monkey_questions2iscorrects.keys())
    intersect_questions = sorted(helm_question_set & monkey_question_set)
    filtered_columns = [
        col for q in intersect_questions
        for col in helm_resmat.columns
        if col[helm_resmat.columns.names.index("input.text")] == q
    ]
    helm_resmat = helm_resmat.loc[:, filtered_columns]
    monkey_questions2iscorrects = {q: monkey_questions2iscorrects[q] for q in intersect_questions}
    
    pass_iat1s = np.array([sum(iscorrects)/len(iscorrects) for iscorrects in monkey_questions2iscorrects.values()])
    num1 = np.sum(pass_iat1s > 0.99)
    numall = len(pass_iat1s)
    percentage = num1 / numall * 100
    print(f"{num1} / {numall} ({percentage:.2f}%)")
    
    # keep_mask = pass_iat1s <= 0.99
    # filtered_questions = np.array(list(monkey_questions2iscorrects.keys()))[keep_mask]
    # monkey_questions2iscorrects = {q: monkey_questions2iscorrects[q] for q in filtered_questions}
    # filtered_columns = [
    #     col for q in filtered_questions
    #     for col in helm_resmat.columns
    #     if col[helm_resmat.columns.names.index("input.text")] == q
    # ]
    # helm_resmat = helm_resmat.loc[:, filtered_columns]
    
    pass_iat1s = np.array([sum(iscorrects)/len(iscorrects) for iscorrects in monkey_questions2iscorrects.values()])
    cache_path = f"pass_iatk_matrix_{monkey_model_name}_{monkey_scenario}.npy"
    if os.path.exists(cache_path):
        pass_iatk_matrix = np.load(cache_path)
    else:
        pass_iatk_matrix = np.stack([
            np.array([
                estimate_success_rate_at_k_per_problem(
                    len(iscorrects),
                    sum(iscorrects),
                    k
                ) for k in range(1, len(iscorrects) + 1)
            ])
            for iscorrects in tqdm(monkey_questions2iscorrects.values())
        ])  # shape: (n_questions, k)
        np.save(cache_path, pass_iatk_matrix)

    n_questions, k = pass_iatk_matrix.shape
    print(pass_iatk_matrix.shape)
    k_arange = np.arange(1, k + 1)
    
    ### 1. least square estimator
    print("1. least square estimator")
    pass_datks = pass_iatk_matrix.mean(0) # shape: (k,)
    neglog_gts = -np.log(pass_datks)
    popt, _ = curve_fit(power_law_func, k_arange, neglog_gts)
    a_est, b_est = popt
    neglog_est_1 = power_law_func(k_arange, a_est, b_est) # shape: (k,)
    
    ### 2. distributional estimator
    print("2. distributional estimator")
    pass_datks_est2 = []
    for k in k_arange:
        pass_datk_est2 = 1 - (-np.log(1- (1 - pass_iat1s) ** k)).mean()
        # pass_datk_est2 = 1 - ((1 - pass_iat1s) ** k).mean()
        pass_datks_est2.append(pass_datk_est2)
    neglog_est_2 = -np.log(np.array(pass_datks_est2))
    
    ### 3. distributional estimator with IRT
    print("3. distributional estimator with IRT")
    specific_model_index = helm_resmat.index.tolist().index(helm_model_name)
    device = "cuda:5"
    data = torch.tensor(helm_resmat.values, dtype=torch.float64, device=device)
    added_data = np.array(list(monkey_questions2iscorrects.values()))
    added_data = torch.from_numpy(added_data).to(device=device).double().T
    data = torch.cat([data, added_data], dim=0)
    n_test_takers, n_items = data.shape
    print(data.shape)
    mean_28 = data[28].mean().item()
    mean_91_onwards = data[91:].mean(dim=1).cpu().numpy()
    mean_values = [mean_28] + list(mean_91_onwards)
    plt.figure(figsize=(8, 4))
    plt.hist(mean_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title("Distribution of Llama38B CTT")
    plt.xlabel("CTT")
    plt.savefig(f"ctt_distri_{monkey_model_name}_{monkey_scenario}.png", dpi=300, bbox_inches='tight')
    
    B = 50000
    n_thetas_nuisance = 150
    optimized_z = []
    thetas_nuisance = torch.randn(n_thetas_nuisance, n_test_takers, device=device, dtype=torch.float64)
    for i in tqdm(range(0, n_items, B)):
        data_batch = data[:, i:i+B]
        current_B = data_batch.shape[1]
        z = torch.randn(current_B, requires_grad=True, device=device, dtype=torch.float64)
        optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        def closure_z():
            optim_z.zero_grad()
            mask = ~torch.isnan(data_batch).expand(n_thetas_nuisance, -1, -1)
            probs = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
            loss = -(Bernoulli(probs=probs[mask]).log_prob(
                data_batch.expand(n_thetas_nuisance, -1, -1)[mask]
            )).mean()
            loss.backward()
            probs_holder['current'] = probs.detach()
            return loss
        z_optimized = trainer([z], optim_z, closure_z)[0].detach()
        optimized_z.append(z_optimized)
    zs = torch.cat(optimized_z)
    # zs = torch.tensor(helm_resmat.columns.get_level_values("z").astype(float), dtype=torch.float, device=device)
    corr = pearsonr(torch.sigmoid(zs).cpu().numpy(), pass_iat1s).statistic
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(6,6))
        plt.scatter(torch.sigmoid(zs).cpu().numpy(), pass_iat1s)
        plt.xlabel("sigmoid(z)", fontsize=20)
        plt.ylabel("Success Rate", fontsize=20)
        plt.title(r'Pearson Correlation: {:.2f}'.format(corr), fontsize=22)
        plt.tick_params(axis="both", labelsize=14)
        plt.plot([0, 1], [0, 1]) 
        plt.xlim(0, 1)
        plt.ylim(0, 1)  
        plt.savefig(f"1pl_sigz_vs_score_corr_{monkey_model_name}_{monkey_scenario}.png", dpi=300)
    
    data_specific_model = data[-1]
    theta = torch.randn((1,), requires_grad=True, device=device, dtype=torch.float64)
    optim_theta = LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_theta():
        optim_theta.zero_grad()
        mask = ~torch.isnan(data_specific_model)
        probs = torch.sigmoid(theta + zs)
        loss = -(Bernoulli(probs=probs[mask]).log_prob(data_specific_model[mask])).mean()
        loss.backward()
        probs_holder['current'] = probs.detach()
        return loss
    theta = trainer([theta], optim_theta, closure_theta)[0].detach()
    probs = torch.sigmoid(theta + zs).cpu().numpy() # shape: (n_questions,)
    
    corr = pearsonr(probs, pass_iat1s).statistic
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(6,6))
        plt.scatter(probs, pass_iat1s)
        plt.xlabel("IRT Probability", fontsize=20)
        plt.ylabel("Success Rate", fontsize=20)
        plt.title(r'Pearson Correlation: {:.2f}'.format(corr), fontsize=22)
        plt.tick_params(axis="both", labelsize=14)
        plt.plot([0, 1], [0, 1]) 
        plt.xlim(0, 1)
        plt.ylim(0, 1)  
        plt.savefig(f"1pl_theta_vs_score_corr_{monkey_model_name}_{monkey_scenario}.png", dpi=300)
        plt.show()
    
    pass_datks_est3 = []
    for k in k_arange:
        pass_datk_est3 = 1 - (-np.log(1- (1 - probs) ** k)).mean()
        pass_datks_est3.append(pass_datk_est3)
    neglog_est_3 = -np.log(np.array(pass_datks_est3))
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(6, 6))
        
        # ground truth: black solid
        plt.loglog(k_arange, neglog_gts,
                linestyle='-',
                color='black',
                linewidth=2,
                label='Ground truth')
        
        # least‐squares estimator: dashed
        plt.loglog(k_arange, neglog_est_1,
                linestyle='--',
                label='Least squares')
        
        # distributional estimator: dashed
        plt.loglog(k_arange, neglog_est_2,
                linestyle='--',
                label='Distributional')
        
        # IRT‐based estimator: dashed
        plt.loglog(k_arange, neglog_est_3,
                linestyle='--',
                label='1PL IRT')
        
        plt.xlabel(r'$k$', fontsize=20)
        plt.ylabel(r'$-\log\bigl(\mathrm{pass}_{\mathcal{D}}@k\bigr)$', fontsize=20)
        plt.tick_params(axis="both", labelsize=14)
        plt.legend(fontsize=14)
        plt.savefig(f"estimator_comparison_{monkey_model_name}_{monkey_scenario}.png", dpi=300)
    
