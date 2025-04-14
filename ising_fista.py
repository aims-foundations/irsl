import pandas as pd
import torch
torch.manual_seed(0)
from tqdm import tqdm
from torch.distributions import Bernoulli
from torchmetrics import AUROC
auroc = AUROC(task="binary")
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, ListedColormap
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import logging
import numpy as np
from datetime import datetime

def split_train_test(total_count, train_percentage):
    train_count = int(total_count * train_percentage)
    indices = torch.randperm(total_count)
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]
    return train_indices, test_indices

def get_lasso_data(data, n_rhos):
    """
    Constructs inputs, labels, rho_seqs for Lasso variable selection.

    Args:
        data (torch.Tensor): Binary matrix of shape (N, P)
        n_rhos (int): Number of rho values to generate

    Returns:
        inputs (torch.Tensor): Tensor of shape (N, P, P)
        labels (torch.Tensor): Tensor of shape (N, n_rhos, P)
        rho_seqs (torch.Tensor): Tensor of shape (n_rhos, P)
    """
    N, P = data.shape
    # Compute rho_max = max(|X^T y|) / n
    rho_min_ratio = 1e-3 if N > P else 1e-2

    subsets, leftovers = [], []
    for i in range(P):
        mask = torch.ones(P, dtype=torch.bool)
        mask[i] = False
        subset = data[:, mask]
        subset = torch.cat((subset, torch.ones(N, 1)), dim=1)
        leftover = data[:, i].unsqueeze(1)
        subsets.append(subset.unsqueeze(2))
        leftovers.append(leftover.unsqueeze(2))
    inputs = torch.cat(subsets, dim=2)   # (N, P, P)
    labels = torch.cat(leftovers, dim=2) # (N, P)
    labels = labels.expand(-1, n_rhos, -1)  # (N, n_rhos, P)

    rho_seqs = []
    for i in range(P):
        labels_ = labels[:, 0, i]
        inputs_ = inputs[:, :, i]
        rho_max = (inputs_.T @ (labels_ - 0.5)).abs().max() / N
        rho_sequence = torch.logspace(
            torch.log10(rho_max),
            torch.log10(rho_max * rho_min_ratio),
            steps=n_rhos
        )
        rho_seqs.append(rho_sequence)
    rho_seqs = torch.stack(rho_seqs, dim=1)  # (n_rhos, P)

    return inputs, labels, rho_seqs

def stage1_variable_selection(args, inputs_train, labels_train, rho_seqs_):
    """
    Performs sparse variable selection using FISTA with bootstrapping and cross-validation.

    Args:
        args: Namespace containing hyperparameters:
            - n_bootstraps (int): Number of bootstrap iterations.
            - n_cv (int): Number of cross-validation folds.
            - cv_percentage (float): Percentage of data used for cross-validation.
            - n_epochs_sparse (int): Number of epochs for sparse training.
            - lr_sparse (float): Learning rate for sparse training.
            - freq_threshold (float): Frequency threshold for variable selection.
            - gpuid (int): GPU ID for computation.
        inputs_train (torch.Tensor): Input tensor of shape (N_train, P, B).
        labels_train (torch.Tensor): Label tensor of shape (N_train, n_rhos, B).
        rho_seqs_ (torch.Tensor): Regularization parameters of shape (n_rhos, B).

    Returns:
        W_freq (torch.Tensor): Frequency of variable selection across bootstraps, shape (P, B).
        W_binary (torch.Tensor): Binary mask of selected variables, shape (P, B).
        avg_auc (float): Average AUC across bootstraps and best-performing rho values.
    """
    def soft_thresholding(x, alpha):
        return torch.sign(x) * torch.clamp(torch.abs(x) - alpha, min=0.0)
    
    N_train, P, B = inputs_train.shape
    n_rhos = labels_train.shape[1]
    W_freq_bootstraps, auc_bootstraps = [], []
    for _ in tqdm(range(args.n_bootstraps), desc="Bootstraps"):
        bootstrap_indices = torch.randint(0, N_train, (N_train,), device=f'cuda:{args.gpuid}')
        boot_inputs_train = inputs_train[bootstrap_indices, :, :]
        boot_labels_train = labels_train[bootstrap_indices, :, :]
        
        auc_cvs = []
        for _ in tqdm(range(args.n_cv), desc="Cross Validation"):
            cv_train_indices, cv_test_indices = split_train_test(N_train, args.cv_percentage)
            cv_inputs_train = boot_inputs_train[cv_train_indices, :, :]
            cv_inputs_test = boot_inputs_train[cv_test_indices, :, :]
            cv_labels_train = boot_labels_train[cv_train_indices, :, :]
            cv_labels_test = boot_labels_train[cv_test_indices, :, :]
            
            W = torch.randn((P, n_rhos, B), device=f'cuda:{args.gpuid}', requires_grad=False)
            pbar = tqdm(range(args.n_epochs_sparse), desc="Sparse Training")
            W_fista = W.clone().detach()
            t = torch.tensor(1.0, dtype=torch.float32, device=f'cuda:{args.gpuid}')
            for epoch in pbar:
                if epoch > 0:
                    prev_loss = loss.detach().clone()
                    prev_W = W.detach().clone()
                
                logits = torch.einsum('ijp,jkp->ikp', cv_inputs_train, W) # shape: (N_train, n_rhos, B)
                probs = torch.sigmoid(logits)
                # TODO: grad calculation might have a bug
                error = (probs - cv_labels_train) / (N_train*args.cv_percentage)
                grad = torch.einsum('ijp,ikp->jkp', cv_inputs_train, error) # shape: (P, n_rhos, B)
                W_fista_new = soft_thresholding(W - args.lr_sparse * grad, args.lr_sparse * rho_seqs_)
                t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t**2)) / 2.0
                momentum = (1 - t) / t_new
                W = (1-momentum)*W_fista_new + momentum*W_fista
                W_fista = W_fista_new
                t = t_new

                loss = -Bernoulli(probs=probs).log_prob(cv_labels_train).mean()
                reg_term = (rho_seqs_ * torch.norm(W, p=1, dim=0)).mean()
                if epoch > 0:
                    d_loss = (prev_loss - loss).item()
                    d_W = torch.norm(prev_W - W, p=2).item() # / W.numel()
                    grad_norm = torch.norm(grad, p=2).item() # / grad.numel()
                    pbar.set_postfix({
                        "loss": loss.item(),
                        "reg_term": reg_term.item(),
                        "d_loss": d_loss,
                        "d_W": d_W,
                        "grad_norm": grad_norm,
                    })
                    eps = 1e-5
                    if d_loss < eps and d_W < eps and grad_norm < eps:
                        break
            
            logits_test = torch.einsum('ijp,jkp->ikp', cv_inputs_test, W) # shape: (N_test, n_rhos, B)
            auc_Bs = []
            for j in tqdm(range(B), desc="Calculate AUC"):
                auc_rhos = []
                for k in range(n_rhos):
                    auc_rho = auroc(
                        torch.sigmoid(logits_test[:,k,j].detach().cpu()),
                        cv_labels_test[:,k,j].detach().cpu(),
                    )
                    auc_rhos.append(auc_rho)
                    auroc.reset()
                auc_rhos = torch.stack(auc_rhos, dim=0) # shape: (n_rhos,)
                auc_Bs.append(auc_rhos)
            auc_Bs = torch.stack(auc_Bs, dim=1) # shape: (n_rhos, B)
            auc_cvs.append(auc_Bs)
            
        auc_cvs = torch.stack(auc_cvs, dim=2) # shape: (n_rhos, B, n_seeds)
        auc_cvs = auc_cvs.mean(dim=2) # shape: (n_rhos, B)
        # pick the largest (first) rho that has highest test AUC
        best_rho_indices = auc_cvs.argmax(dim=0) # shape: (B,)
        auc_bootstrap = auc_cvs[best_rho_indices, torch.arange(B)].mean()
        auc_bootstraps.append(auc_bootstrap.item())
        W = W[:, best_rho_indices, torch.arange(B)] # shape: (P, B)
        W_binary = (W != 0).float()
        W_freq_bootstraps.append(W_binary)
    
    avg_auc = sum(auc_bootstraps)/len(auc_bootstraps)
    W_freq_bootstraps = torch.stack(W_freq_bootstraps, dim=2) # shape: (P, B, n_bootstraps)
    W_freq = torch.mean(W_freq_bootstraps, dim=2) # shape: (P, B)
    W_freq = torch.maximum(W_freq, W_freq.T)
    W_binary = (W_freq > args.freq_threshold) # shape: (P, B)
    
    return W_freq, W_binary, avg_auc

def stage2_refit(args, inputs_train, labels_train_single, W_binary):
    """
    Perform logistic regression refitting using masked inputs and binary weight mask.

    Args:
        inputs_train (Tensor): Input tensor of shape (N_train, P, B)
        labels_train_single (Tensor): Label tensor of shape (N_train, B)
        W_binary (Tensor): Binary mask tensor of shape (P, B)
        args: Arguments with fields gpuid, lr_refit, and n_epochs_refit

    Returns:
        Tensor: Refit weight matrix of shape (P, B)
    """
    def refit_trainer(parameters, optim, closure, epochs, eps = 1e-5):
        pbar = tqdm(range(epochs), desc="Refit Epoch")
        loss = closure()
        for _ in pbar:
            prev_para = [p.clone().detach() for p in parameters]
            prev_loss = loss.clone().detach()
            loss = optim.step(closure)
            d_loss = (prev_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(prev_para, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            pbar.set_postfix({
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "d_parameter": d_parameters,
                "d_loss": d_loss
            })
            if d_loss < eps and d_parameters < eps and grad_norm < eps:
                break
        return parameters

    N_train, P, B = inputs_train.shape
    W_binary_expand = W_binary.unsqueeze(0).expand(N_train, -1, -1) # shape: (N_train, P, B)
    inputs_train_masked = inputs_train * W_binary_expand
    W = torch.randn((P, B), device=f'cuda:{args.gpuid}', requires_grad=True)
    optimizer = torch.optim.LBFGS([W], lr=args.lr_refit, max_iter=1000, history_size=10, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        logits = torch.einsum('ijp,jp->ip', inputs_train_masked, W) # shape: (N_train, B)
        probs = torch.sigmoid(logits)
        loss = -Bernoulli(probs=probs).log_prob(labels_train_single).mean()
        loss.backward()
        return loss
    W = refit_trainer([W], optimizer, closure, args.n_epochs_refit)[0]  # (P, B)
    return (W * W_binary).detach()

def stage3_gibbs_sampling(args, data_test_observe, W):
    """
    Perform Gibbs sampling for missing data imputation in a given dataset.

    Args:
        args: Object containing Gibbs sampling parameters (`n_burn_in`, `n_samples`, `gpuid`).
        data_test_observe (torch.Tensor): Tensor of observed data of shape (N_test, P_observe).
        W (torch.Tensor): Weight matrix of shape (P, P) for energy computation.

    Returns:
        torch.Tensor: Imputed data of shape (N_test, P_impute), averaged over `n_samples`.
    """
    N_test, P_observe = data_test_observe.shape
    P = W.shape[0]
    P_impute = P - P_observe

    data_test_impute = torch.randint(0, 2, (N_test, P_impute), device=f'cuda:{args.gpuid}', dtype=torch.float32)
    data_test = torch.cat((data_test_observe, data_test_impute), dim=1)
    
    data_test_sum = torch.zeros_like(data_test)
    for t in tqdm(range(args.n_burn_in + args.n_samples), desc="Gibbs Sampling"):
        rand_index = torch.randint(P_observe, P, (1,)).item()
        data_test_new = data_test.clone()
        data_test_new[:, rand_index] = 1 - data_test[:, rand_index]
        unnorm_prob_old = 0.5 * torch.einsum("bi,ij,bj->b", data_test, W, data_test)
        unnorm_prob_new = 0.5 * torch.einsum("bi,ij,bj->b", data_test_new, W, data_test_new)
        accept_prob = torch.minimum(torch.ones_like(unnorm_prob_old), torch.exp(unnorm_prob_new - unnorm_prob_old))
        rand_vals = torch.rand(N_test, device=f'cuda:{args.gpuid}')
        accept_mask = rand_vals < accept_prob
        data_test[accept_mask, rand_index] = data_test_new[accept_mask, rand_index]
        if t >= args.n_burn_in:
            data_test_sum += data_test

    return data_test_sum[:, P_observe:] / args.n_samples
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--resmat_path", type=str, required=True, help="Path to response matrix")
    parser.add_argument("--gpuid", type=int, default=4, help="GPU ID")
    parser.add_argument("--train_percentage", type=float, default=0.8, help="Percentage of train set for train-test split")
    parser.add_argument("--step_size", type=int, default=8192, help="Node batch size")
    # Stage 1 - Sparse Training (Variable Selection)
    parser.add_argument("--n_rhos", type=int, default=10, help="Number of regularization parameters to try")
    parser.add_argument("--n_bootstraps", type=int, default=3, help="Number of bootstraps")
    parser.add_argument("--freq_threshold", type=float, default=0.9, help="Feature frequency threshold for boothstrap")
    parser.add_argument("--n_cv", type=int, default=3, help="Number of cross-validation runs")
    parser.add_argument("--cv_percentage", type=float, default=0.8, help="Percentage of train set for cross-validation split")
    parser.add_argument("--n_epochs_sparse", type=int, default=1000, help="Number of sparse training epochs")
    parser.add_argument("--lr_sparse", type=float, default=0.1, help="Learning rate for sparse training")
    # Stage 2 - Refit (Debias)
    parser.add_argument("--n_epochs_refit", type=int, default=1000, help="Number of refit epochs")
    parser.add_argument("--lr_refit", type=float, default=0.01, help="Learning rate for refit")
    # Stage 3 - Gibbs Sampling (Ising Inference)
    parser.add_argument("--gs_percentage", type=float, default=0.8, help="Percentage of items used as observed")
    parser.add_argument("--n_burn_in", type=int, default=1000, help="Number of burn-in steps")
    parser.add_argument("--n_samples", type=int, default=50000, help="Number of Gibbs samples")
    args = parser.parse_args()

    # create output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"result/ising/{args.resmat_path.split('.')[0]}_fista_torch_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # create log file
    log_path = f"{output_dir}/log.txt"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # read response matrix and fill missing data with column majority, turn into tensor
    input_df = pd.read_csv(args.resmat_path)
    input_df = input_df.apply(lambda col: col.fillna(1 if (col == 1).sum() > (col == 0).sum() else 0))
    input_df = input_df[np.random.permutation(input_df.columns)]
    data = torch.tensor(input_df.values, dtype=torch.float32)
    N, P = data.shape
    logging.info(f"{data.shape}")

    # get lasso data
    inputs, labels, rho_seqs = get_lasso_data(data, args.n_rhos)
    
    W_freqs, W_binarys, Ws = [], [], []
    train_indices, test_indices = split_train_test(N, args.train_percentage)
    for i in tqdm(range(0, P, args.step_size), desc="Batch"):
        B = min(args.step_size, P - i)
        inputs_ = inputs[:, :, i:i+B].to(f'cuda:{args.gpuid}') # shape: (N, P, B)
        labels_ = labels[:, :, i:i+B].to(f'cuda:{args.gpuid}') # shape: (N, n_rhos, B)
        rho_seqs_ = rho_seqs[:, i:i+B].to(f'cuda:{args.gpuid}') # shape: (n_rhos, B)

        inputs_train = inputs_[train_indices, :, :]
        inputs_test = inputs_[test_indices, :, :]
        labels_train = labels_[train_indices, :, :]
        labels_train_single = labels_train[:, 0, :]
        labels_test_single = labels_[test_indices, :, :][:, 0, :]

        # Stage 1 - Sparse Training (Variable Selection)
        W_freq, W_binary, valid_auc = stage1_variable_selection(args, inputs_train, labels_train, rho_seqs_)
        logging.info(f"stage 1 valid auc: {valid_auc}")
        W_freqs.append(W_freq)
        W_binarys.append(W_binary)
        non_zero_count = (W_binary != 0).sum().item()
        total_count = W_binary.numel()
        logging.info(f"stage 1 non-zero elements: {non_zero_count} / {total_count} ({100*non_zero_count/total_count:.2f}%)")
    
        # Stage 2 - Refit (Debias)
        W = stage2_refit(args, inputs_train, labels_train_single, W_binary)
        Ws.append(W)
        logits_train = torch.einsum('ijp,jp->ip', inputs_train, W) # shape: (N, B)
        train_auc = auroc(torch.sigmoid(logits_train), labels_train_single)
        logits_test = torch.einsum('ijp,jp->ip', inputs_test, W)
        test_auc = auroc(torch.sigmoid(logits_test), labels_test_single)
        logging.info(f"stage 2 train auc: {train_auc}; test auc: {test_auc}")
        non_zero_count = (W != 0).sum().item()
        total_count = W.numel()
        logging.info(f"stage 2 non-zero elements: {non_zero_count} / {total_count} ({100*non_zero_count/total_count:.2f}%)")
        
        # Stage 3 - Gibbs Sampling (Ising Inference)
        P_observe = int(P*args.gs_percentage)
        data_test_observe = data[test_indices, :][:, :P_observe].to(f'cuda:{args.gpuid}')
        data_test_inpute_true = data[test_indices, :][:, :P_observe].to(f'cuda:{args.gpuid}')
        data_test_inpute_probs = stage3_gibbs_sampling(args, data_test_observe, W)
        gs_auc = auroc(data_test_inpute_probs, data_test_inpute_true)
        logging.info(f"stage 3 gibbs sampling auc: {gs_auc}")

    # save result
    W_freqs = torch.cat(W_freqs, dim=1).cpu() # shape: (P, P)
    W_binarys = torch.cat(W_binarys, dim=1).cpu() # shape: (P, P)
    Ws = torch.cat(Ws, dim=1).cpu() # shape: (P, P)
    torch.save(W_freqs, f'{output_dir}/W_freq.pt')
    torch.save(W_binarys, f'{output_dir}/W_binary.pt')
    torch.save(Ws, f'{output_dir}/W.pt')
    W_df = pd.DataFrame(Ws.numpy(), columns=input_df.columns)
    W_df.to_csv(f'{output_dir}/W.csv', index=False)

    # visualize W
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        cmap_w_freq = LinearSegmentedColormap.from_list("W_freq_cmap", ["white", "blue"])
        norm_w_freq = Normalize(vmin=0, vmax=1)
        cmap_w_binary = ListedColormap(['white', 'blue'])
        lower_perc, upper_perc = np.percentile(Ws.numpy(), [2, 98])
        norm_w = TwoSlopeNorm(vmin=lower_perc, vcenter=0, vmax=upper_perc)
        cmap_w = LinearSegmentedColormap.from_list("W_cmap", ["red", "white", "blue"])
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        im0 = axes[0].imshow(W_freqs.numpy(), cmap=cmap_w_freq, norm=norm_w_freq)
        axes[0].set_title("W_freq")
        ticks_freq = np.linspace(0, 1, 7)
        cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, ticks=ticks_freq)
        cbar0.ax.set_yticklabels([f"{t:.2f}" for t in ticks_freq])
        im1 = axes[1].imshow(W_binarys.numpy(), cmap=cmap_w_binary)
        axes[1].set_title("W_binary")
        im2 = axes[2].imshow(Ws.numpy(), cmap=cmap_w, norm=norm_w)
        axes[2].set_title("W")
        ticks_w = np.linspace(lower_perc, upper_perc, 7)
        cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=ticks_w)
        cbar2.ax.set_yticklabels([f"{t:.2f}" for t in ticks_w])
        plt.savefig(f"{output_dir}/W.png", dpi=300)