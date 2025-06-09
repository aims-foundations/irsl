import pandas as pd
from tqdm import tqdm
import torch
import pickle
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import numpy as np
import wandb
np.random.seed(0)
torch.manual_seed(42)
import os
from sklearn.metrics import mean_squared_error
from pathlib import Path
from sklearn.metrics import roc_auc_score
import argparse
from sklearn.linear_model import LogisticRegression
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore")

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
    wandb.init(project="monkey_generalize_scaleup_no_monkey_z")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    path = args.path

    device = "cuda:0"
    B = 50000
    method = "diff_split" # "55randomsplit"
    scenarios = scenario2benchmark.keys()

    output_dir = f"../../result/monkey_generalize_scaleup_{method}_logi_regr"
    os.makedirs(output_dir, exist_ok=True)
    
    stem = Path(path).stem # e.g. "pythia-12b_lsat_qa"
    scenario_name = next((s for s in scenarios if stem.endswith(f"_{s}")), None)
    monkey_model_name = stem[: -(len(scenario_name) + 1)] # remove "_lsat_qa" from the end to get the model name
    
    benchmark_name = scenario2benchmark.get(scenario_name)
    print(f"\nmodel={monkey_model_name}, scenario={scenario_name}, benchmark={benchmark_name}")
    # if os.path.exists(f"{output_dir}/nonlog_{monkey_model_name}_{scenario_name}.png"):
    #     continue

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
    # get monkey resmat
    monkey_seqs = [torch.tensor(l, dtype=torch.float64) for l in monkey_questions2iscorrects.values()]
    monkey_data = pad_sequence(monkey_seqs, padding_value=float('nan')).to(device)
    n_test_takers, n_items = monkey_data.shape
    print(f"all_data.shape: {monkey_data.shape}")
    
    # two split methods
    if method == "diff_split":
        temperature = 0.1  # Tune this: lower = more extreme bias, higher = softer bias
        split_idx = n_items // 2
        probs = ( (helm_zs - helm_zs.min() + 1e-6).pow(1.0/temperature) )
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
    helm_train_zs        = helm_zs[train_idxs]
    helm_test_zs         = helm_zs[test_idxs]
    
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
    
    ### 2. distributional estimator
    train_pass_datk_est2s = np.array([cal_passdatk(train_pass_iat1s, k) for k in k_arange])
    train_neglog_est2s = -np.log(train_pass_datk_est2s)
    
    ### 3. LR estimator
    train_monkey_data = monkey_data[:, train_idxs]
    train_monkey_data_expanded = train_monkey_data.reshape(-1).cpu().numpy()
    monkey_train_zs_expanded = helm_train_zs.repeat(train_monkey_data.shape[0]).cpu().numpy().reshape(-1, 1)
    mask = ~np.isnan(train_monkey_data_expanded)
    X_train = monkey_train_zs_expanded[mask]
    y_train = train_monkey_data_expanded[mask].astype(int)
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
    lr.fit(X_train, y_train)
    
    train_probs = lr.predict_proba(X_train.reshape(-1, 1))[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"Train ROC-AUC: {train_auc:.4f}")
    
    test_monkey_data = monkey_data[:, test_idxs]
    test_flat = test_monkey_data.reshape(-1).cpu().numpy()
    helm_test_zs_expanded = helm_test_zs.repeat(test_monkey_data.shape[0]).cpu().numpy().reshape(-1, 1)
    mask_test = ~np.isnan(test_flat)
    X_test = helm_test_zs_expanded[mask_test]
    y_test = test_flat[mask_test].astype(int)
    test_probs = lr.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    print(f" Test ROC-AUC: {test_auc:.4f}")
    breakpoint()
    
    train_probs = lr.predict_proba(helm_train_zs.cpu().numpy().reshape(-1, 1))[:, 1]
    test_probs  = lr.predict_proba(helm_test_zs.cpu().numpy().reshape(-1, 1))[:, 1]
    train_pass_datk_est3s = np.array([cal_passdatk(train_probs, k) for k in k_arange])
    train_neglog_est3s    = -np.log(train_pass_datk_est3s)
    test_pass_datk_est3s  = np.array([cal_passdatk(test_probs,  k) for k in k_arange])
    test_neglog_est3s     = -np.log(test_pass_datk_est3s)
    
    # plot passat1 correlation with probs
    all_values = np.concatenate([helm_train_zs.cpu().numpy() , helm_test_zs.cpu().numpy()])
    small, large = all_values.min(), all_values.max()
    helm_z_range = np.linspace(small - 1, large + 1, 200).reshape(-1, 1)
    lr_probs_plot = lr.predict_proba(helm_z_range)[:, 1]
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(6,6))
        plt.scatter(helm_train_zs.cpu().numpy() , train_pass_iat1s, c="blue", label="Train")
        plt.scatter(helm_test_zs.cpu().numpy() , test_pass_iat1s, c="red", label="Test")
        plt.plot(helm_z_range, lr_probs_plot, linestyle='-', linewidth=2, label="LR fit")
        plt.xlabel("HELM $z$", fontsize=20)
        plt.ylabel("Pass i at 1", fontsize=20)
        plt.tick_params(axis="both", labelsize=14)
        plt.xlim(small-1, large+1)
        plt.ylim(0, 1)
        plt.legend(fontsize=16)
        plt.savefig(f"{output_dir}/passat1_vs_helmz_{monkey_model_name}_{scenario_name}.png", dpi=300)

    # save result
    results_dict = {
        "train_pass_datk_gts": train_pass_datk_gts,
        "test_pass_datk_gts": test_pass_datk_gts,
        "train_pass_datk_est2s": train_pass_datk_est2s,
        "train_pass_datk_est3s": train_pass_datk_est3s,
        "test_pass_datk_est3s": test_pass_datk_est3s,
    }
    with open(f"{output_dir}/monkey_scaleup_data_{monkey_model_name}_{scenario_name}.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    # plot passdatk vs k, only x-axis use log scale
    mse_train_dist  = mean_squared_error(train_pass_datk_gts, train_pass_datk_est2s)
    mse_test_dist  = mean_squared_error(test_pass_datk_gts, train_pass_datk_est2s)
    mse_train_rasch = mean_squared_error(train_pass_datk_gts, train_pass_datk_est3s)
    mse_test_rasch  = mean_squared_error(test_pass_datk_gts, test_pass_datk_est3s)
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(k_arange, train_pass_datk_gts, linestyle='-', color='blue', linewidth=2, label='Train GT', alpha=0.5)
        ax.semilogx(k_arange, test_pass_datk_gts, linestyle='-', color='red',   linewidth=2, label='Test GT', alpha=0.5)
        ax.semilogx(k_arange, train_pass_datk_est2s, linestyle='--', color='blue', label=f'Train Distri (Train MSE={mse_train_dist:.2e}, Test MSE={mse_test_dist:.2e})', alpha=0.5)
        ax.semilogx(k_arange, train_pass_datk_est3s, linestyle=':',  color='blue', label=f'Train LR (MSE={mse_train_rasch:.2e})', alpha=0.5)
        ax.semilogx(k_arange, test_pass_datk_est3s,  linestyle=':',  color='red',  label=f'Test LR (MSE={mse_test_rasch:.2e})', alpha=0.5)
        ax.set_xlabel(r'$k$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{pass}_{\mathcal{D}}@k$', fontsize=20)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=12, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/nonlog_{monkey_model_name}_{scenario_name}.png", dpi=300, bbox_inches="tight")
