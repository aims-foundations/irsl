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
from pathlib import Path
import glob
from huggingface_hub import snapshot_download
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore")

def irt_formula(theta, z, guess=0):
    return guess + (1-guess)*torch.sigmoid(theta[:, None] + z[None, :])

def irt_formula_nuisance(theta, z, guess=0):
    return guess + (1-guess)*torch.sigmoid(theta[:, :, None] + z[None, None, :])

probs_holder = {'current': None}
def trainer(parameters, optim, closure, n_iter=100):
    pbar = tqdm(range(n_iter))
    for iteration in pbar:
        if iteration > 0:
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
            pbar.set_postfix({"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss, "d_probs": d_probs})
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5 and d_probs < 1e-5:
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

def cal_passiatk_matrix(max_len, train_monkey_questions2iscorrects):
    return np.stack([
        np.pad(
            np.array([
                estimate_success_rate_at_k_per_problem(len(iscorrects), sum(iscorrects), k)
                for k in range(1, len(iscorrects) + 1)
            ]),
            # pad on the right with edge values (i.e. repeat the last element)
            (0, max_len - len(iscorrects)),
            mode='edge',
        )
        for iscorrects in tqdm(train_monkey_questions2iscorrects.values())
    ]) # shape: (n_questions, k)

def cal_passdatk(passiat1, k):
    return (1 - (1 - passiat1) ** k).mean()
                              
def power_law_func(k, a, b):
    return a * k ** (-b)

def linear_func(z, w, b):
    return w * z + b
        
benchmark2scenario = {
    "lite": ["legalbench", "math", "commonsense", "med_qa", "gsm"],
    "mmlu": ["mmlu"],
    "classic": ["bbq", "lsat_qa", "legal_support"]
}
scenario2benchmark = {
    **{
        scenario: benchmark_name
        for benchmark_name, scenario_list in benchmark2scenario.items()
        for scenario in scenario_list
    },
    'harm_bench': 'safety',
    'gsm': 'lite',
}

if __name__ == "__main__":
    device = "cuda:2"
    B = 50000
    method = "diff_split" # "55randomsplit"
    
    local_dir = snapshot_download(
        repo_id="stair-lab/monkey_queries",
        repo_type="dataset"
    )
    scenarios = scenario2benchmark.keys()
    paths = []
    for scenario in scenarios:
        paths.extend(glob.glob(f"{local_dir}/*{scenario}.json"))

    output_dir = f"../../result/monkey_generalize_scaleup_{method}"
    os.makedirs(output_dir, exist_ok=True)
    
    for path in tqdm(paths)   :        
        stem = Path(path).stem # e.g. "pythia-12b_lsat_qa"
        scenario_name = next((s for s in scenarios if stem.endswith(f"_{s}")), None)
        monkey_model_name = stem[: -(len(scenario_name) + 1)] # remove "_lsat_qa" from the end to get the model name
        benchmark_name = scenario2benchmark.get(scenario_name)
        print(f"\nmodel={monkey_model_name}, scenario={scenario_name}, benchmark={benchmark_name}")
        if os.path.exists(f"{output_dir}/nonlog_{monkey_model_name}_{scenario_name}.png"):
            continue

        # load monkey data
        monkey_dataset = pd.read_json(path)
        monkey_questions2iscorrects = {row["question"]: row["is_corrects"] for _, row in monkey_dataset.iterrows()}
        print(f"len(monkey_questions2iscorrects): {len(monkey_questions2iscorrects)}")
        lengths = [len(l) for l in monkey_questions2iscorrects.values()]
        print(f"set(lengths): {set(lengths)}")
        max_len = max(lengths)
        print(f"max_len: {max_len}")
        
        # load HELM data
        pkl_name = "results_with_z" if benchmark_name != "safety" else "results_with_z_harmbench"
        with open(f"/lfs/skampere1/0/sttruong/deval/data/gather_helm_data/{pkl_name}.pkl", "rb") as f:
            helm_resmat = pickle.load(f)
        helm_resmat = helm_resmat.loc[:, ~helm_resmat.columns.get_level_values("z").isna()] # drop z=nan
        helm_resmat = helm_resmat.loc[:, 
                    (helm_resmat.columns.get_level_values("benchmark") == benchmark_name) &
                    (helm_resmat.columns.get_level_values("scenario")  == scenario_name)
                ] # filter benchamrk & scenario
        helm_resmat = helm_resmat.dropna(how="all", axis=0) # drop empty rows
        if scenario_name == "legal_support":
            cols_df = helm_resmat.columns.to_frame(index=False)
            cols_df["input.text"] = (
                cols_df["input.text"].astype(str)
                + cols_df["references"].astype(str)
            )
            helm_resmat.columns = pd.MultiIndex.from_frame(cols_df)
        helm_resmat = helm_resmat.loc[:, ~helm_resmat.columns.get_level_values("input.text").duplicated(keep=False)] # drop duplicate questions
        print(f"helm_resmat.shape: {helm_resmat.shape}")
            
        # get intersect questions
        intersect_mask = helm_resmat.columns.get_level_values("input.text").isin(monkey_questions2iscorrects)
        helm_resmat = helm_resmat.loc[:, intersect_mask]
        intersect_questions = helm_resmat.columns.get_level_values("input.text").tolist()
        print(f"intersect: {len(intersect_questions)}")
        monkey_questions2iscorrects = {q: monkey_questions2iscorrects[q] for q in intersect_questions}
        
        # get helm_zs
        helm_zs = torch.tensor(helm_resmat.columns.get_level_values("z").astype(float), dtype=torch.float64, device=device)
        # calibrate for monkey_zs
        helm_data = torch.tensor(helm_resmat.values, dtype=torch.float64, device=device)
        monkey_seqs = [torch.tensor(l, dtype=torch.float64) for l in monkey_questions2iscorrects.values()]
        monkey_data = pad_sequence(monkey_seqs, padding_value=float('nan')).to(device)
        all_data = torch.cat([helm_data, monkey_data], dim=0)
        n_test_takers, n_items = all_data.shape
        print(f"all_data.shape: {all_data.shape}")
        n_thetas_nuisance = 150
        monkey_zs = []
        thetas_nuisance = torch.randn(n_thetas_nuisance, n_test_takers, device=device, dtype=torch.float64)
        for i in tqdm(range(0, n_items, B)):
            all_data_batch = all_data[:, i:i+B]
            current_B = all_data_batch.shape[1]
            z = torch.randn(current_B, requires_grad=True, device=device, dtype=torch.float64)
            optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
            def closure_z():
                optim_z.zero_grad()
                mask = ~torch.isnan(all_data_batch).expand(n_thetas_nuisance, -1, -1)
                probs = irt_formula_nuisance(thetas_nuisance, z)
                loss = -(Bernoulli(probs=probs[mask]).log_prob(
                    all_data_batch.expand(n_thetas_nuisance, -1, -1)[mask]
                )).mean()
                loss.backward()
                probs_holder['current'] = probs.detach()
                return loss
            monkey_z = trainer([z], optim_z, closure_z)[0].detach()
            monkey_zs.append(monkey_z)
        monkey_zs = torch.cat(monkey_zs)

        # two split methods
        if method == "diff_split":
            temperature = 0.5  # Tune this: lower = more extreme bias, higher = softer bias
            split_idx = n_items // 2
            probs = ( (helm_zs.max() - helm_zs + 1e-6).pow(1.0 / temperature) )
            probs /= probs.sum()
            train_idxs = torch.multinomial(probs, split_idx, replacement=False).tolist()  
            test_idxs = [i for i in list(range(n_items)) if i not in train_idxs]
        elif method == "55randomsplit":
            indices = torch.randperm(n_items)
            split = n_items // 2
            train_idxs = indices[:split].tolist()
            test_idxs  = indices[split:].tolist()

        # index train & test
        train_helm_resmat = helm_resmat.iloc[:, train_idxs]
        test_helm_resmat  = helm_resmat.iloc[:, test_idxs]
        train_monkey_questions2iscorrects = {intersect_questions[i]: monkey_questions2iscorrects[intersect_questions[i]] for i in train_idxs}
        test_monkey_questions2iscorrects = {intersect_questions[i]: monkey_questions2iscorrects[intersect_questions[i]] for i in test_idxs}
        monkey_train_zs_true = monkey_zs[train_idxs]
        monkey_test_zs_true  = monkey_zs[test_idxs]
        helm_train_zs        = helm_zs[train_idxs]
        helm_test_zs         = helm_zs[test_idxs]
        
        # fit: monkey_zs = w * helm_zs + b
        params, _ = curve_fit(linear_func, helm_train_zs.cpu().numpy(), monkey_train_zs_true.cpu().numpy())
        w, b = params
        monkey_train_zs_pred = torch.tensor(linear_func(helm_train_zs.cpu().numpy(), w, b), dtype=torch.float64, device=device)
        monkey_test_zs_pred = torch.tensor(linear_func(helm_test_zs.cpu().numpy(), w, b), dtype=torch.float64, device=device)
        # plot helm_zs vs monkey_zs
        plt.figure()
        x_line = np.linspace(helm_train_zs.cpu().numpy().min(), helm_train_zs.cpu().numpy().max(), 100)
        y_line = w * x_line + b
        plt.plot(x_line, y_line, linestyle='-', label='Fitted line')
        plt.scatter(helm_train_zs.cpu().numpy(), monkey_train_zs_true.cpu().numpy(), marker='o', label='Train true')
        plt.scatter(helm_train_zs.cpu().numpy(), monkey_train_zs_pred.cpu().numpy(), marker='x', label='Train predicted')
        plt.scatter(helm_test_zs.cpu().numpy(), monkey_test_zs_true.cpu().numpy(), marker='o', label='Test true')
        plt.scatter(helm_test_zs.cpu().numpy(), monkey_test_zs_pred.cpu().numpy(), marker='x', label='Test predicted')
        plt.xlabel('helm_zs')
        plt.ylabel('monkey_zs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/helm_zs_vs_monkey_zs_{monkey_model_name}_{scenario_name}.png", dpi=300)
        
        # calculate passiat1 and passiatkmatrix
        train_pass_iat1s = np.array([sum(iscorrects)/len(iscorrects) for iscorrects in train_monkey_questions2iscorrects.values()])
        train_pass_iatk_matrix = cal_passiatk_matrix(max_len, train_monkey_questions2iscorrects)        
        test_pass_iat1s = np.array([sum(iscorrects)/len(iscorrects) for iscorrects in test_monkey_questions2iscorrects.values()])
        test_pass_iatk_matrix = cal_passiatk_matrix(max_len, test_monkey_questions2iscorrects)
        print(f"train_pass_iatk_matrix.shape: {train_pass_iatk_matrix.shape}")
        k_arange = np.arange(1, max_len + 1)
        
        ### 0. gt
        train_pass_datk_gts = train_pass_iatk_matrix.mean(0) # shape: (max_len,)
        train_neglog_gts = -np.log(train_pass_datk_gts)
        test_pass_datk_gts = test_pass_iatk_matrix.mean(0) # shape: (max_len,)
        test_neglog_gts = -np.log(test_pass_datk_gts)
        
        ### 1. least square estimator
        popt, _ = curve_fit(power_law_func, k_arange, train_neglog_gts)
        a_est, b_est = popt
        train_neglog_est1s = power_law_func(k_arange, a_est, b_est) # shape: (max_len,)
        train_pass_datk_est1s = np.exp(-train_neglog_est1s)
        
        ### 2. distributional estimator
        train_pass_datk_est2s = np.array([cal_passdatk(train_pass_iat1s, k) for k in k_arange])
        train_neglog_est2s = -np.log(train_pass_datk_est2s)
        
        ### 3. Rasch estimator
        # fit theta
        train_all_data = all_data[:, train_idxs]
        train_all_data_expanded = train_all_data.reshape(-1)
        monkey_train_zs_expanded = monkey_train_zs_true.repeat_interleave(train_all_data.shape[0])
        theta = torch.randn((1,), requires_grad=True, device=device, dtype=torch.float64)
        optim_theta = LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        def closure_theta():
            optim_theta.zero_grad()
            mask = ~torch.isnan(train_all_data_expanded)[None, :]
            probs = irt_formula(theta, monkey_train_zs_expanded)
            loss = -(Bernoulli(probs=probs[mask]).log_prob(train_all_data_expanded[None, :][mask])).mean()
            loss.backward()
            probs_holder['current'] = probs.detach()
            return loss
        theta = trainer([theta], optim_theta, closure_theta)[0].detach()
        print(f"theta: {theta.item()}")
        # calculate train probs
        # train_probs = irt_formula(theta, monkey_train_zs_true).reshape(-1).cpu().numpy() 
        # train_probs shape: (n_items_train,). should replace monkey_train_zs_true with monkey_train_zs_pred to access the performance of 3-parameter model (theta, w, b)
        train_probs = irt_formula(theta, monkey_train_zs_pred).reshape(-1).cpu().numpy() 
        train_pass_datk_est3s = np.array([cal_passdatk(train_probs, k) for k in k_arange])
        train_neglog_est3s = -np.log(train_pass_datk_est3s)
        # calculate test probs
        test_probs = irt_formula(theta, monkey_test_zs_pred).reshape(-1).cpu().numpy()
        test_pass_datk_est3s = np.array([cal_passdatk(test_probs, k) for k in k_arange])
        test_neglog_est3s = -np.log(test_pass_datk_est3s)
        # plot passat1 correlation with irt probs
        corr_train = pearsonr(train_probs, train_pass_iat1s).statistic
        corr_test = pearsonr(test_probs, test_pass_iat1s).statistic
        all_values = np.concatenate([train_probs, test_probs, train_pass_iat1s, test_pass_iat1s])
        small, large = all_values.min(), all_values.max()
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            plt.figure(figsize=(6,6))
            plt.scatter(train_probs, train_pass_iat1s, c="blue")
            plt.scatter(test_probs, test_pass_iat1s, c="red")
            plt.xlabel("Sigmoid (theta + z)", fontsize=20)
            plt.ylabel("Pass i at 1", fontsize=20)
            plt.title(r'Train Corr: {:.2f}, Test Corr: {:.2f}'.format(corr_train, corr_test), fontsize=22)
            plt.tick_params(axis="both", labelsize=14)
            plt.plot([small, large], [small, large]) 
            plt.xlim(small, large)
            plt.ylim(small, large)
            plt.savefig(f"{output_dir}/prob_corr_{monkey_model_name}_{scenario_name}.png", dpi=300)

        # save result
        results_dict = {
            "train_pass_datk_gts": train_pass_datk_gts,
            "test_pass_datk_gts": test_pass_datk_gts,
            "train_pass_datk_est1s": train_pass_datk_est1s,
            "train_pass_datk_est2s": train_pass_datk_est2s,
            "train_pass_datk_est3s": train_pass_datk_est3s,
            "test_pass_datk_est3s": test_pass_datk_est3s,
        }
        with open(f"{output_dir}/monkey_scaleup_data_{monkey_model_name}_{scenario_name}.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # plog -log(passdatk) vs k, log-log space
        mse_train_ls    = mean_squared_error(train_neglog_gts, train_neglog_est1s)
        mse_train_dist  = mean_squared_error(train_neglog_gts, train_neglog_est2s)
        mse_train_rasch = mean_squared_error(train_neglog_gts, train_neglog_est3s)
        mse_test_ls     = mean_squared_error(test_neglog_gts, train_neglog_est1s)
        mse_test_dist   = mean_squared_error(test_neglog_gts, train_neglog_est2s)
        mse_test_rasch  = mean_squared_error(test_neglog_gts, test_neglog_est3s)
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # ————— Left subplot: training curves —————
            ax = axes[0]
            ax.loglog(k_arange, train_neglog_gts, linestyle='-', color='black', linewidth=2, label='Ground truth')
            ax.loglog(k_arange, train_neglog_est1s, linestyle='--', label=f'Least squares (MSE={mse_train_ls:.2e})')
            ax.loglog(k_arange, train_neglog_est2s, linestyle='--', label=f'Distributional (MSE={mse_train_dist:.2e})')
            ax.loglog(k_arange, train_neglog_est3s, linestyle='--', label=f'Rasch (MSE={mse_train_rasch:.2e})')
            ax.set_xlabel(r'$k$', fontsize=20)
            ax.set_ylabel(r'$-\log\bigl(\mathrm{pass}_{\mathcal{D}}@k\bigr)$', fontsize=20)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=14)
            ax.set_title('Train', fontsize=16)
            # ————— Right subplot: test curves —————
            ax = axes[1]
            ax.loglog(k_arange, test_neglog_gts, linestyle='-', color='black', linewidth=2, label='Ground truth')
            ax.loglog(k_arange, train_neglog_est1s, linestyle='--', label=f'Least squares (MSE={mse_test_ls:.2e})')
            ax.loglog(k_arange, train_neglog_est2s, linestyle='--', label=f'Distributional (MSE={mse_test_dist:.2e})')
            ax.loglog(k_arange, test_neglog_est3s, linestyle='--', label=f'Rasch (MSE={mse_test_rasch:.2e})')
            ax.set_xlabel(r'$k$', fontsize=20)
            ax.set_ylabel(r'$-\log\bigl(\mathrm{pass}_{\mathcal{D}}@k\bigr)$', fontsize=20)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=14)
            ax.set_title('Test', fontsize=16)
            # save
            fig.tight_layout()
            fig.savefig(f"{output_dir}/monkey_generalize_{monkey_model_name}_{scenario_name}.png", dpi=300, bbox_inches="tight")

        # plot passdatk vs k, only x-axis use log scale
        mse_train_dist  = mean_squared_error(train_pass_datk_gts, train_pass_datk_est2s)
        mse_train_rasch = mean_squared_error(train_pass_datk_gts, train_pass_datk_est3s)
        mse_test_dist   = mean_squared_error(test_pass_datk_gts, train_pass_datk_est2s)
        mse_test_rasch  = mean_squared_error(test_pass_datk_gts, test_pass_datk_est3s)
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # ————— Left subplot: training curves —————
            ax = axes[0]
            ax.semilogx(k_arange, train_pass_datk_gts, linestyle='-', color='black', linewidth=2, label='Ground truth')
            ax.semilogx(k_arange, train_pass_datk_est2s, linestyle='--', label=f'Distributional (MSE={mse_train_dist:.2e})')
            ax.semilogx(k_arange, train_pass_datk_est3s, linestyle='--', label=f'Rasch (MSE={mse_train_rasch:.2e})')
            ax.set_xlabel(r'$k$', fontsize=20)
            ax.set_ylabel(r'$\mathrm{pass}_{\mathcal{D}}@k$', fontsize=20)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=14)
            ax.set_title('Train', fontsize=16)
            # ————— Right subplot: test curves —————
            ax = axes[1]
            ax.semilogx(k_arange, test_pass_datk_gts, linestyle='-', color='black', linewidth=2, label='Ground truth')
            ax.semilogx(k_arange, train_pass_datk_est2s, linestyle='--', label=f'Distributional (MSE={mse_test_dist:.2e})')
            ax.semilogx(k_arange, test_pass_datk_est3s, linestyle='--', label=f'Rasch (MSE={mse_test_rasch:.2e})')
            ax.set_xlabel(r'$k$', fontsize=20)
            ax.set_ylabel(r'$\mathrm{pass}_{\mathcal{D}}@k$', fontsize=20)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=14)
            ax.set_title('Test', fontsize=16)
            # save
            fig.tight_layout()
            fig.savefig(f"{output_dir}/nonlog_{monkey_model_name}_{scenario_name}.png", dpi=300, bbox_inches="tight")