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
torch.manual_seed(42)
from scipy.optimize import curve_fit
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.kernel_ridge import KernelRidge
from pathlib import Path
import glob
from huggingface_hub import snapshot_download
import warnings
warnings.filterwarnings("ignore")

def irt_formula(theta, z, guess):
    return guess + (1-guess)*torch.sigmoid(theta[:, None] + z[None, :])
def irt_formula_nuisance(theta, z, guess):
    return guess + (1-guess)*torch.sigmoid(theta[:, :, None] + z[None, None, :])
scenario2guess = {
    'bbq': 1/3,
    'commonsense': 0.25,
    'gsm': 0,
    'harm_bench': 0,
    'legalbench': 0.2,
    'lsat_qa': 0.2,
    'math': 0,
    'med_qa': 0.25,
    'mmlu': 0.25,
}

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

def cal_passdatk(passiat1, k):
    passiatk = 1 - (1 - passiat1) ** k
    passdatk = passiatk.mean()
    return passdatk
                              
def power_law_func(k, a, b):
    return a * k ** (-b)

def linear_func(z, w, b):
    return w * z + b

if __name__ == "__main__":
    device = "cuda:7"
    B = 50000
    method = "diff_split"
    # method = "55randomsplit"
    
    # ScalingIntelligence/monkey_business
    monkey_business_list = [
        f"hf://datasets/ScalingIntelligence/monkey_business/GSM8K_{monkey_model}.json"
        for monkey_model in ["Llama-3-8B-Instruct", "Llama-3-70B-Instruct"]
    ]
    
    # Rylan query
    # rylan_query_list = [
    #     f"hf://datasets/stair-lab/monkey_queries/rylan_query/GSM8K_Pythia_6.9B.json",
    #     f"hf://datasets/stair-lab/monkey_queries/rylan_query/GSM8K_Pythia_12B.json",
    # ]
    # rylan_query_list = glob.glob("data/rylan_monkey/GSM8K*")
    local_dir = snapshot_download(
        repo_id="stair-lab/monkey_queries",
        repo_type="dataset"
    )
    rylan_query_list = glob.glob(
        os.path.join(local_dir, "rylan_query", "GSM8K*")
    )
    
    # we query
    monkey_model_names_1 = ["Meta-Llama-3-8B-Instruct", "pythia-12b", "pythia-6.9b"]
    benchmark_scenarios_1 = {
        "lite": ["legalbench", "math", "commonsense", "med_qa"],
        "mmlu": ["mmlu"],
        "classic": ["bbq", "lsat_qa"] # , "legal_support"
    }
    we_query_list_1 = [
        f"hf://datasets/stair-lab/monkey_queries/{monkey_model_name}_{scenario_name}.json"
        for monkey_model_name in monkey_model_names_1
        for scenario_list in benchmark_scenarios_1.values()
        for scenario_name in scenario_list
    ]
    monkey_model_names_2 = [ 
        "Mistral-7B-v0.1",
        "gemma-3-27b-it",
        "Qwen3-8B",
        "Qwen3-32B",
        "Qwen3-14B",
        "DeepSeek-V2-Lite-Chat",
        "DeepSeek-R1-Distill-Llama-8B",
    ]
    benchmark_scenarios_2 = {
        "lite": ["gsm"],
    }
    we_query_list_2 = [
        f"hf://datasets/stair-lab/monkey_queries/{monkey_model_name}_{scenario_name}.json"
        for monkey_model_name in monkey_model_names_2
        for scenario_list in benchmark_scenarios_2.values()
        for scenario_name in scenario_list
    ]
    
    # harmbench
    harmbench_model_names = [
        "GraySwanAI-Llama-3-8B-Instruct-RR",
        "claude-3-5-sonnet-20240620",
        "cygnet",
        "gpt-4o-mini",
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "meta-llama-Meta-Llama-3-8B-Instruct",
        "gpt-4o",
        "claude-3-opus-20240229",
    ]
    harmbench_list = [
        f"hf://datasets/stair-lab/monkey_queries/{harmbench_model_name}_harm_bench.json"
        for harmbench_model_name in harmbench_model_names
    ]
    
    scenario_to_benchmark = {
        **{
            scenario: benchmark_name
            for benchmark_name, scenario_list in benchmark_scenarios_1.items()
            for scenario in scenario_list
        },
        **{
            scenario: benchmark_name
            for benchmark_name, scenario_list in benchmark_scenarios_2.items()
            for scenario in scenario_list
        },
        'harm_bench': 'safety',
        'gsm': 'lite',
    }
    
    all_list = we_query_list_1
    # monkey_business_list + rylan_query_list + we_query_list_1 + we_query_list_2 + harmbench_list
    for path in all_list:
        stem = Path(path).stem # e.g. "pythia-12b_lsat_qa"
        monkey_model_name, scenario_name = stem.split("_", 1)
        if monkey_model_name == "GSM8K":
            monkey_model_name = scenario_name
            scenario_name = "gsm"

        benchmark_name = scenario_to_benchmark.get(scenario_name)
        print(f"\nmodel={monkey_model_name}, scenario={scenario_name}, benchmark={benchmark_name}")
        # guess = scenario2guess[scenario_name]
        guess = 0
        monkey_dataset = pd.read_json(path)
        # monkey_dataset = pd.read_json(f"hf://datasets/stair-lab/monkey_queries/{monkey_model_name}_{scenario_name}.json")
        
        pkl_name = "results_with_z" if benchmark_name != "safety" else "results_with_z_harmbench"
        with open(f"/lfs/skampere1/0/sttruong/deval/data/gather_helm_data/{pkl_name}.pkl", "rb") as f:
            helm_resmat_full = pickle.load(f)
        keep_cols = ~helm_resmat_full.columns.get_level_values("z").isna()
        helm_resmat_full = helm_resmat_full.loc[:, keep_cols]
        
        output_dir = f"result/monkey_generalize_scaleup_{method}"
        os.makedirs(output_dir, exist_ok=True)
        
        monkey_questions2iscorrects = {row["question"]: row["is_corrects"] for _, row in monkey_dataset.iterrows()}
        print(f"len(monkey_questions2iscorrects): {len(monkey_questions2iscorrects)}")
        lengths = [len(l) for l in monkey_questions2iscorrects.values()]
        print(f"set(lengths): {set(lengths)}")
        
        helm_resmat = helm_resmat_full.loc[:, helm_resmat_full.columns.get_level_values("benchmark") == benchmark_name]
        helm_resmat = helm_resmat.loc[:, helm_resmat.columns.get_level_values("scenario") == scenario_name]
        helm_resmat = helm_resmat[~helm_resmat.isna().all(axis=1)]
        print(f"helm_resmat.shape: {helm_resmat.shape}")
        helm_questions = list(helm_resmat.columns.get_level_values("input.text"))

        helm_question_set = set(helm_questions)
        monkey_question_set = set(monkey_questions2iscorrects.keys())
        intersect_questions = sorted(helm_question_set & monkey_question_set)
        print(f"interset: {len(intersect_questions)}")
        filtered_columns = [
            col for q in intersect_questions
            for col in helm_resmat.columns
            if col[helm_resmat.columns.names.index("input.text")] == q
        ]
        helm_resmat = helm_resmat.loc[:, filtered_columns]
        monkey_questions2iscorrects = {q: monkey_questions2iscorrects[q] for q in intersect_questions}
        
        helm_zs = torch.tensor(helm_resmat.columns.get_level_values("z").astype(float), dtype=torch.float, device=device)
        
        data = torch.tensor(helm_resmat.values, dtype=torch.float64, device=device)
        # added_data = np.array(list(monkey_questions2iscorrects.values()))
        added_lists = list(monkey_questions2iscorrects.values())
        max_len = max(len(l) for l in added_lists)
        padded = [
            l + [np.nan] * (max_len - len(l))
            for l in added_lists
        ]
        added_data = np.array(padded, dtype=float)
        added_data = torch.from_numpy(added_data).to(device=device).double().T
        all_data = torch.cat([data, added_data], dim=0)
        
        # n_test_takers, n_items = added_data.shape
        # print(f"added_data.shape: {added_data.shape}")
        # n_thetas_nuisance = 150
        # optimized_z = []
        # thetas_nuisance = torch.randn(n_thetas_nuisance, n_test_takers, device=device, dtype=torch.float64)
        # for i in tqdm(range(0, n_items, B)):
        #     added_data_batch = added_data[:, i:i+B]
        #     current_B = added_data_batch.shape[1]
        #     z = torch.randn(current_B, requires_grad=True, device=device, dtype=torch.float64)
        #     optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        #     def closure_z():
        #         optim_z.zero_grad()
        #         mask = ~torch.isnan(added_data_batch).expand(n_thetas_nuisance, -1, -1)
        #         probs = irt_formula_nuisance(thetas_nuisance, z, guess)
        #         loss = -(Bernoulli(probs=probs[mask]).log_prob(
        #             added_data_batch.expand(n_thetas_nuisance, -1, -1)[mask]
        #         )).mean()
        #         loss.backward()
        #         probs_holder['current'] = probs.detach()
        #         return loss
        #     z_optimized = trainer([z], optim_z, closure_z)[0].detach()
        #     optimized_z.append(z_optimized)
        # zs = torch.cat(optimized_z)
              
        n_test_takers, n_items = all_data.shape
        print(f"all_data.shape: {all_data.shape}")
        n_thetas_nuisance = 150
        optimized_z = []
        thetas_nuisance = torch.randn(n_thetas_nuisance, n_test_takers, device=device, dtype=torch.float64)
        for i in tqdm(range(0, n_items, B)):
            all_data_batch = all_data[:, i:i+B]
            current_B = all_data_batch.shape[1]
            z = torch.randn(current_B, requires_grad=True, device=device, dtype=torch.float64)
            optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
            def closure_z():
                optim_z.zero_grad()
                mask = ~torch.isnan(all_data_batch).expand(n_thetas_nuisance, -1, -1)
                probs = irt_formula_nuisance(thetas_nuisance, z, guess)
                loss = -(Bernoulli(probs=probs[mask]).log_prob(
                    all_data_batch.expand(n_thetas_nuisance, -1, -1)[mask]
                )).mean()
                loss.backward()
                probs_holder['current'] = probs.detach()
                return loss
            z_optimized = trainer([z], optim_z, closure_z)[0].detach()
            optimized_z.append(z_optimized)
        zs = torch.cat(optimized_z)

        if method == "diff_split":
            # split = n_items // 2
            # sorted_desc = torch.argsort(helm_zs, descending=True)
            # train_idxs = sorted_desc[:split].tolist()
            # test_idxs  = sorted_desc[split:].tolist()
            
            temperature = 0.5  # Tune this: lower = more extreme bias, higher = softer bias
            split_idx = n_items // 2
            eps = 1e-6
            weights = ((helm_zs.max() - helm_zs) + eps) ** (1.0 / temperature)
            probs = weights / weights.sum()
            train_idxs = torch.multinomial(probs, split_idx, replacement=False)
            all_idxs = torch.arange(n_items)
            mask = torch.zeros(n_items, dtype=torch.bool)
            mask[train_idxs] = True
            test_idxs = all_idxs[~mask]

        elif method == "55randomsplit":
            indices = torch.randperm(n_items)
            split = int(0.5 * n_items)
            train_idxs = indices[:split].tolist()
            test_idxs  = indices[split:].tolist()

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
        
        # X_train = helm_train_zs.cpu().numpy().reshape(-1, 1)
        # y_train = train_zs.cpu().numpy()
        # X_test  = helm_test_zs.cpu().numpy().reshape(-1, 1)
        # kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
        # kr.fit(X_train, y_train)
        # test_zs = kr.predict(X_test)
        
        plt.figure()
        plt.scatter(helm_train_zs.cpu().numpy(), train_zs.cpu().numpy(), marker='o', label='Train data')
        x_line = np.linspace(helm_train_zs.cpu().numpy().min(), helm_train_zs.cpu().numpy().max(), 100)
        y_line = w * x_line + b
        x_line = x_line.reshape(-1, 1)
        # y_line = kr.predict(x_line)
        plt.plot(x_line, y_line, linestyle='-', label='Fitted line')
        plt.scatter(helm_test_zs.cpu().numpy(), test_zs_true.cpu().numpy(), marker='x', label='Test true')
        plt.scatter(helm_test_zs.cpu().numpy(), test_zs, marker='^', label='Test predicted')
        plt.xlabel('helm_zs')
        plt.ylabel('monkey_zs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/helm_zs_vs_monkey_zs_{monkey_model_name}_{scenario_name}.png", dpi=300)
        
        train_pass_iat1s = np.array([sum(iscorrects)/len(iscorrects) for iscorrects in train_monkey_questions2iscorrects.values()])
        train_pass_iatk_matrix = np.stack([
            np.pad(
                np.array([
                    estimate_success_rate_at_k_per_problem(
                        len(iscorrects),
                        sum(iscorrects),
                        k
                    )
                    for k in range(1, len(iscorrects) + 1)
                ]),
                # pad on the right with edge values (i.e. repeat the last element)
                (0, max_len - len(iscorrects)),
                mode='edge'
            )
            for iscorrects in tqdm(train_monkey_questions2iscorrects.values())
        ]) # shape: (n_questions, k)
        _, k = train_pass_iatk_matrix.shape
        print(f"train_pass_iatk_matrix.shape: {train_pass_iatk_matrix.shape}")
        k_arange = np.arange(1, k + 1)
        
        test_pass_iat1s = np.array([sum(iscorrects)/len(iscorrects) for iscorrects in test_monkey_questions2iscorrects.values()])
        test_pass_iatk_matrix = np.stack([
            np.pad(
                np.array([
                    estimate_success_rate_at_k_per_problem(
                        len(iscorrects),
                        sum(iscorrects),
                        k
                    )
                    for k in range(1, len(iscorrects) + 1)
                ]),
                (0, max_len - len(iscorrects)),
                mode='edge'
            )
            for iscorrects in tqdm(test_monkey_questions2iscorrects.values())
        ]) # shape: (n_questions, k)
        
        ### 1. least square estimator
        train_pass_datks = train_pass_iatk_matrix.mean(0) # shape: (k,)
        train_neglog_gts = -np.log(train_pass_datks)
        popt, _ = curve_fit(power_law_func, k_arange, train_neglog_gts)
        a_est, b_est = popt
        train_neglog_est_1 = power_law_func(k_arange, a_est, b_est) # shape: (k,)
        
        test_pass_datks = test_pass_iatk_matrix.mean(0) # shape: (k,)
        test_neglog_gts = -np.log(test_pass_datks)
        
        ### 2. distributional estimator
        train_pass_datks_est2 = []
        for k in k_arange:
            train_pass_datk_est2 = cal_passdatk(train_pass_iat1s, k)
            train_pass_datks_est2.append(train_pass_datk_est2)
        train_neglog_est_2 = -np.log(np.array(train_pass_datks_est2))
        
        ### 3. distributional estimator with IRT        
        train_data_expanded = added_data[:, train_idxs].reshape(-1)  
        train_zs_expanded = train_zs.repeat_interleave(added_data.shape[0])
        theta = torch.randn((1,), requires_grad=True, device=device, dtype=torch.float64)
        optim_theta = LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        def closure_theta():
            optim_theta.zero_grad()
            mask = ~torch.isnan(train_data_expanded)[None, :]
            probs = irt_formula(theta, train_zs_expanded, guess)
            loss = -(Bernoulli(probs=probs[mask]).log_prob(train_data_expanded[None, :][mask])).mean()
            loss.backward()
            probs_holder['current'] = probs.detach()
            return loss
        theta = trainer([theta], optim_theta, closure_theta)[0].detach()
        
        print(f"theta: {theta.item()}")
        train_probs = irt_formula(theta, train_zs, guess).reshape(-1).cpu().numpy() # shape: (n_questions,)
        train_pass_datks_est3 = []
        for k in k_arange:
            train_pass_datk_est3 = cal_passdatk(train_probs, k)
            train_pass_datks_est3.append(train_pass_datk_est3)
        train_neglog_est_3 = -np.log(np.array(train_pass_datks_est3))
        
        test_zs = torch.tensor(test_zs, dtype=torch.float64, device=device)
        test_probs = irt_formula(theta, test_zs, guess).reshape(-1).cpu().numpy() # shape: (n_questions,)
        test_pass_datks_est3 = []
        for k in k_arange:
            test_pass_datk_est3 =cal_passdatk(test_probs, k)
            test_pass_datks_est3.append(test_pass_datk_est3)
        test_neglog_est_3 = -np.log(np.array(test_pass_datks_est3))
        
        corr_train = pearsonr(train_probs, train_pass_iat1s).statistic
        corr_test = pearsonr(test_probs, test_pass_iat1s).statistic
        all_values = np.concatenate([train_probs, test_probs, train_pass_iat1s, test_pass_iat1s])
        small, large = all_values.min(), all_values.max()
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            plt.figure(figsize=(6,6))
            plt.scatter(train_probs, train_pass_iat1s, c="blue")
            plt.scatter(test_probs, test_pass_iat1s, c="red")
            plt.xlabel("sigmoid(theta+z)", fontsize=20)
            plt.ylabel("Success Rate", fontsize=20)
            plt.title(r'Train Corr: {:.2f}, Test Corr: {:.2f}'.format(corr_train, corr_test), fontsize=22)
            plt.tick_params(axis="both", labelsize=14)
            plt.plot([small, large], [small, large]) 
            plt.xlim(small, large)
            plt.ylim(small, large)
            plt.savefig(f"{output_dir}/prob_corr_{monkey_model_name}_{scenario_name}.png", dpi=300)

        results_dict = {
            "train_neglog_gts": train_neglog_gts,
            "test_neglog_gts": test_neglog_gts,
            "train_neglog_est_1": train_neglog_est_1,
            "train_neglog_est_2": train_neglog_est_2,
            "train_neglog_est_3": train_neglog_est_3,
            "test_neglog_est_3": test_neglog_est_3,
        }
        
        mse_train_ls    = mean_squared_error(train_neglog_gts, train_neglog_est_1)
        mse_train_dist  = mean_squared_error(train_neglog_gts, train_neglog_est_2)
        mse_train_rasch = mean_squared_error(train_neglog_gts, train_neglog_est_3)
        mse_test_ls     = mean_squared_error(test_neglog_gts, train_neglog_est_1)
        mse_test_dist   = mean_squared_error(test_neglog_gts, train_neglog_est_2)
        mse_test_rasch  = mean_squared_error(test_neglog_gts, test_neglog_est_3)
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
                    label=f'Least squares (MSE={mse_train_ls:.2e})')
            ax.loglog(k_arange, train_neglog_est_2,
                    linestyle='--',
                    label=f'Distributional (MSE={mse_train_dist:.2e})')
            ax.loglog(k_arange, train_neglog_est_3,
                    linestyle='--',
                    label=f'Rasch (MSE={mse_train_rasch:.2e})')
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
            ax.loglog(k_arange, train_neglog_est_1,
                    linestyle='--',
                    label=f'Least squares (MSE={mse_test_ls:.2e})')
            ax.loglog(k_arange, train_neglog_est_2,
                    linestyle='--',
                    label=f'Distributional (MSE={mse_test_dist:.2e})')
            ax.loglog(k_arange, test_neglog_est_3,
                    linestyle='--',
                    label=f'Rasch (MSE={mse_test_rasch:.2e})')
            ax.set_xlabel(r'$k$', fontsize=20)
            ax.set_ylabel(r'$-\log\bigl(\mathrm{pass}_{\mathcal{D}}@k\bigr)$', fontsize=20)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=14)
            ax.set_title('Test', fontsize=16)

            fig.tight_layout()
            fig.savefig(f"{output_dir}/monkey_generalize_{monkey_model_name}_{scenario_name}.png", dpi=300, bbox_inches="tight")

            with open(f"{output_dir}/monkey_scaleup_data_{monkey_model_name}_{scenario_name}.pkl", "wb") as f:
                pickle.dump(results_dict, f)
        
        mse_train_dist  = mean_squared_error(train_pass_datks, train_pass_datks_est2)
        mse_train_rasch = mean_squared_error(train_pass_datks, train_pass_datks_est3)
        mse_test_dist   = mean_squared_error(test_pass_datks, train_pass_datks_est2)
        mse_test_rasch  = mean_squared_error(test_pass_datks, test_pass_datks_est3)
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # ————— Left subplot: training curves —————
            ax = axes[0]
            ax.semilogx(k_arange, train_pass_datks,
                        linestyle='-',
                        color='black',
                        linewidth=2,
                        label='Ground truth')
            ax.semilogx(k_arange, np.array(train_pass_datks_est2),
                        linestyle='--',
                        label=f'Distributional (MSE={mse_train_dist:.2e})')
            ax.semilogx(k_arange, np.array(train_pass_datks_est3),
                        linestyle='--',
                        label=f'Rasch (MSE={mse_train_rasch:.2e})')
            ax.set_xlabel(r'$k$', fontsize=20)
            ax.set_ylabel(r'$\mathrm{pass}_{\mathcal{D}}@k$', fontsize=20)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=14)
            ax.set_title('Train', fontsize=16)

            # ————— Right subplot: test curves —————
            ax = axes[1]
            ax.semilogx(k_arange, test_pass_datks,
                        linestyle='-',
                        color='black',
                        linewidth=2,
                        label='Ground truth')
            ax.semilogx(k_arange, np.array(train_pass_datks_est2),
                        linestyle='--',
                        label=f'Distributional (MSE={mse_test_dist:.2e})')
            ax.semilogx(k_arange, np.array(test_pass_datks_est3),
                        linestyle='--',
                        label=f'Rasch (MSE={mse_test_rasch:.2e})')
            ax.set_xlabel(r'$k$', fontsize=20)
            ax.set_ylabel(r'$\mathrm{pass}_{\mathcal{D}}@k$', fontsize=20)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=14)
            ax.set_title('Test', fontsize=16)

            fig.tight_layout()
            fig.savefig(f"{output_dir}/nonlog_{monkey_model_name}_{scenario_name}.png", dpi=300, bbox_inches="tight")