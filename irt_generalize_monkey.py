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
import random
random.seed(42)

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

def linear_func(z, w, b):
    return w * z + b

if __name__ == "__main__":
    monkey_model_name = "Llama-3-8B-Instruct" # "Llama-3-70B-Instruct"# "Pythia-6.9B"
    monkey_scenario = "GSM8K" # "GSM8K"
    helm_filename = "lite_gsm_results" # "classic_math_results" # "classic_gsm_results" 
    helm_model_name = "meta/llama-3-8b" # "eleutherai/pythia-6.9b" "meta/llama-3-70b"
    device = "cuda:5"
    
    monkey_dataset = pd.read_json(f"hf://datasets/ScalingIntelligence/monkey_business/{monkey_scenario}_{monkey_model_name}.json")
    monkey_questions2iscorrects = {
        row["question"]: row["is_corrects"]
        for _, row in monkey_dataset.iterrows()
    }
    with open(f"/lfs/skampere1/0/sttruong/deval/gather_helm_data/{helm_filename}.pkl", "rb") as f:
        helm_resmat = pickle.load(f)
    helm_questions = list(helm_resmat.columns.get_level_values("input.text"))

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
    
    helm_zs = torch.tensor(helm_resmat.columns.get_level_values("z").astype(float), dtype=torch.float, device=device)
    
    data = torch.tensor(helm_resmat.values, dtype=torch.float64, device=device)
    added_data = np.array(list(monkey_questions2iscorrects.values()))
    added_data = torch.from_numpy(added_data).to(device=device).double().T
    data = torch.cat([data, added_data], dim=0)
    n_test_takers, n_items = data.shape
    print(data.shape)
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

    n = zs.size(0)
    split = n // 2
    sorted_desc = torch.argsort(zs, descending=True)
    train_idxs = sorted_desc[:split].tolist()
    test_idxs  = sorted_desc[split:].tolist()

    train_questions = [intersect_questions[i] for i in train_idxs]
    test_questions  = [intersect_questions[i] for i in test_idxs]
    train_filtered_columns = [
        col for q in train_questions
        for col in helm_resmat.columns
        if col[helm_resmat.columns.names.index("input.text")] == q
    ]
    train_helm_resmat = helm_resmat.loc[:, train_filtered_columns]
    test_filtered_columns = [
        col for q in test_questions
        for col in helm_resmat.columns
        if col[helm_resmat.columns.names.index("input.text")] == q
    ]
    test_helm_resmat = helm_resmat.loc[:, test_filtered_columns]
    train_monkey_questions2iscorrects = {q: monkey_questions2iscorrects[q] for q in train_questions}
    test_monkey_questions2iscorrects = {q: monkey_questions2iscorrects[q] for q in test_questions}
    
    train_zs = zs[train_idxs]
    test_zs_true = zs[test_idxs]
    helm_train_zs = helm_zs[train_idxs]
    helm_test_zs = helm_zs[test_idxs]
    params, _ = curve_fit(linear_func, helm_train_zs.cpu().numpy(), train_zs.cpu().numpy())
    w, b = params
    test_zs = linear_func(helm_test_zs.cpu().numpy(), w, b)
    
    plt.figure()
    plt.scatter(helm_train_zs.cpu().numpy(), train_zs.cpu().numpy(), marker='o', label='Train data')
    x_line = np.linspace(helm_train_zs.cpu().numpy().min(), helm_train_zs.cpu().numpy().max(), 100)
    y_line = w * x_line + b
    plt.plot(x_line, y_line, linestyle='-', label='Fitted line')
    plt.scatter(helm_test_zs.cpu().numpy(), test_zs_true.cpu().numpy(), marker='x', label='Test true')
    plt.scatter(helm_test_zs.cpu().numpy(), test_zs, marker='^', label='Test predicted')
    plt.xlabel('helm_zs')
    plt.ylabel('monkey_zs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"helm_zs_vs_monkey_zs.png", dpi=300)
    
    train_pass_iat1s = np.array([sum(iscorrects)/len(iscorrects) for iscorrects in train_monkey_questions2iscorrects.values()])
    cache_path = f"train_pass_iatk_matrix_{monkey_model_name}_{monkey_scenario}.npy"
    if os.path.exists(cache_path):
        train_pass_iatk_matrix = np.load(cache_path)
    else:
        train_pass_iatk_matrix = np.stack([
            np.array([
                estimate_success_rate_at_k_per_problem(
                    len(iscorrects),
                    sum(iscorrects),
                    k
                ) for k in range(1, len(iscorrects) + 1)
            ])
            for iscorrects in tqdm(train_monkey_questions2iscorrects.values())
        ])  # shape: (n_questions, k)
        np.save(cache_path, train_pass_iatk_matrix)
    _, k = train_pass_iatk_matrix.shape
    print(train_pass_iatk_matrix.shape)
    k_arange = np.arange(1, k + 1)
    
    cache_path = f"test_pass_iatk_matrix_{monkey_model_name}_{monkey_scenario}.npy"
    if os.path.exists(cache_path):
        test_pass_iatk_matrix = np.load(cache_path)
    else:
        test_pass_iatk_matrix = np.stack([
            np.array([
                estimate_success_rate_at_k_per_problem(
                    len(iscorrects),
                    sum(iscorrects),
                    k
                ) for k in range(1, len(iscorrects) + 1)
            ])
            for iscorrects in tqdm(test_monkey_questions2iscorrects.values())
        ])  # shape: (n_questions, k)
        np.save(cache_path, test_pass_iatk_matrix)
    
    ### 1. least square estimator
    print("1. least square estimator")
    train_pass_datks = train_pass_iatk_matrix.mean(0) # shape: (k,)
    train_neglog_gts = -np.log(train_pass_datks)
    popt, _ = curve_fit(power_law_func, k_arange, train_neglog_gts)
    a_est, b_est = popt
    train_neglog_est_1 = power_law_func(k_arange, a_est, b_est) # shape: (k,)
    
    test_pass_datks = test_pass_iatk_matrix.mean(0) # shape: (k,)
    test_neglog_gts = -np.log(test_pass_datks)
    
    # ### 2. distributional estimator
    # print("2. distributional estimator")
    # pass_datks_est2 = []
    # for k in k_arange:
    #     pass_datk_est2 = 1 - (-np.log(1- (1 - pass_iat1s) ** k)).mean()
    #     # pass_datk_est2 = 1 - ((1 - pass_iat1s) ** k).mean()
    #     pass_datks_est2.append(pass_datk_est2)
    # neglog_est_2 = -np.log(np.array(pass_datks_est2))
    
    ### 3. distributional estimator with IRT
    print("3. distributional estimator with IRT")
    specific_model_index = helm_resmat.index.tolist().index(helm_model_name)
    
    train_data_specific_model = data[-1][train_idxs]
    theta = torch.randn((1,), requires_grad=True, device=device, dtype=torch.float64)
    optim_theta = LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_theta():
        optim_theta.zero_grad()
        mask = ~torch.isnan(train_data_specific_model)
        probs = torch.sigmoid(theta + train_zs)
        loss = -(Bernoulli(probs=probs[mask]).log_prob(train_data_specific_model[mask])).mean()
        loss.backward()
        probs_holder['current'] = probs.detach()
        return loss
    theta = trainer([theta], optim_theta, closure_theta)[0].detach()
    train_probs = torch.sigmoid(theta + train_zs).cpu().numpy() # shape: (n_questions,)
    train_pass_datks_est3 = []
    for k in k_arange:
        train_pass_datk_est3 = 1 - (-np.log(1- (1 - train_probs) ** k)).mean()
        train_pass_datks_est3.append(train_pass_datk_est3)
    train_neglog_est_3 = -np.log(np.array(train_pass_datks_est3))
    
    test_zs = torch.tensor(test_zs, dtype=torch.float64, device=device)
    test_probs = torch.sigmoid(theta + test_zs).cpu().numpy() # shape: (n_questions,)
    test_pass_datks_est3 = []
    for k in k_arange:
        test_pass_datk_est3 = 1 - (-np.log(1- (1 - test_probs) ** k)).mean()
        test_pass_datks_est3.append(test_pass_datk_est3)
    test_neglog_est_3 = -np.log(np.array(test_pass_datks_est3))
    
    
with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # ————— Left subplot: training curves —————
    ax = axes[0]
    ax.loglog(k_arange, train_neglog_gts,
              linestyle='-',
              color='black',
              linewidth=2,
              label='Ground truth')
    ax.loglog(k_arange, train_neglog_est_1,
              linestyle='--',
              label='Least squares')
    ax.loglog(k_arange, train_neglog_est_3,
              linestyle='--',
              label='1PL IRT')
    ax.set_xlabel(r'$k$', fontsize=20)
    ax.set_ylabel(r'$-\log\bigl(\mathrm{pass}_{\mathcal{D}}@k\bigr)$', fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)
    ax.set_title('Train', fontsize=16)

    # ————— Right subplot: test curves —————
    ax = axes[1]
    ax.loglog(k_arange, test_neglog_gts,
              linestyle='-',
              color='black',
              linewidth=2,
              label='Ground truth')
    ax.loglog(k_arange, train_neglog_est_1,  # reuse LS fit
              linestyle='--',
              label='Least squares')
    ax.loglog(k_arange, test_neglog_est_3,
              linestyle='--',
              label='1PL IRT')
    ax.set_xlabel(r'$k$', fontsize=20)
    ax.set_ylabel(r'$-\log\bigl(\mathrm{pass}_{\mathcal{D}}@k\bigr)$', fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)
    ax.set_title('Test', fontsize=16)

    fig.tight_layout()
    fig.savefig(f"generalize_estimator_comparison_{monkey_model_name}_{monkey_scenario}.png", dpi=300)
