import pandas as pd
import torch
from tqdm import tqdm
from torch.distributions import Bernoulli
from torchmetrics import AUROC
auroc = AUROC(task="binary")
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
gpuid = 5
n_rhos = 10
eps = 1e-5

train_percentage = 0.86
step_size = 8192

# stage 1
n_bootstraps = 10
n_CV = 10
CV_percentage = 0.8
n_epochs = 10000
lr = 0.1
freq_threshold = 0.9

# stage 2
n_epochs_refit = 10000

# data preprocess
file_name = 'gsm_hard_easy_200.csv'
output_dir = f"result/ising/{file_name.split('.')[0]}_ista_torch"
os.makedirs(output_dir, exist_ok=True)
input_df = pd.read_csv(file_name)
def fill_by_majority(col):
    ones_count = (col == 1).sum()
    zeros_count = (col == 0).sum()
    majority = 1 if ones_count > zeros_count else 0
    return col.fillna(majority)
filled_df = input_df.apply(fill_by_majority)

data = torch.tensor(filled_df.values, dtype=torch.float32)
N, P = data.shape
print(data.shape)

subsets = []
leftovers = []
for i in range(P):
    # Mask to remove i-th column
    mask = torch.ones(P, dtype=torch.bool)
    mask[i] = False
    subset = data[:, mask] # shape: (N, P-1)
    # add a one column to the subset
    subset = torch.cat((subset, torch.ones(N, 1)), dim=1) # shape: (N, P)
    leftover = data[:, i].unsqueeze(1) # shape: (N, 1)

    subsets.append(subset.unsqueeze(2)) # shape: (N, P-1, 1)
    leftovers.append(leftover.unsqueeze(2)) # shape: (N, 1, 1)

inputs = torch.cat(subsets, dim=2) # shape: (N, P-1, P)
labels = torch.cat(leftovers, dim=2) # shape: (N, P)
labels = labels.expand(-1, n_rhos, -1) # shape: (N, n_rhos, P)

# Compute rho_max = max(|X^T y|) / n
rho_min_ratio = 1e-3 if N > P else 1e-2
rho_seqs = []
max_rhos = []
min_rhos = []
for i in range(P):
    labels_ = labels[:, 0, i] # shape: (N,)
    inputs_ = inputs[:, :, i] # shape: (N, P-1)
    rho_max = inputs_.T @ (labels_-0.5) # shape: (P-1,)
    rho_max = torch.max(torch.abs(rho_max))/N
    rho_sequence = torch.logspace(
        start=torch.log10(rho_max),
        end=torch.log10(rho_max * rho_min_ratio),
        steps=n_rhos,
    )
    rho_seqs.append(rho_sequence)
    max_rhos.append(rho_max)
    min_rhos.append(rho_max * rho_min_ratio)
rho_seqs = torch.stack(rho_seqs, dim=1) # shape: (n_rhos, P)
max_rhos = torch.stack(max_rhos, dim=0)
min_rhos = torch.stack(min_rhos, dim=0)

def soft_thresholding(x, alpha):
    return torch.sign(x) * torch.clamp(torch.abs(x) - alpha, min=0.0)

def trainer(parameters, optim, closure, epochs):
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
        grad_norm = torch.norm(W.grad, p=2).item()
        pbar.set_postfix({
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "d_parameter": d_parameters,
            "d_loss": d_loss
        })
        if d_loss < eps and d_parameters < eps and grad_norm < eps:
            break

    return parameters

with open(f"{output_dir}/print.txt", "w") as f:
    N_train = int(N*train_percentage)
    indices = torch.randperm(N)
    train_indices = indices[:N_train]
    test_indices = indices[N_train:]
    N_CV_train = int(N_train*CV_percentage)
    Ws = []
    for i in tqdm(range(0, P, step_size), desc="Batch"):
        B = min(step_size, P - i)
        inputs_ = inputs[:, :, i:i+B].to(f'cuda:{gpuid}') # shape: (N, P-1, B)
        labels_ = labels[:, :, i:i+B].to(f'cuda:{gpuid}') # shape: (N, n_rhos, B)
        rho_seqs_ = rho_seqs[:, i:i+B].to(f'cuda:{gpuid}') # shape: (n_rhos, B)

        inputs_train = inputs_[train_indices, :, :]
        inputs_test = inputs_[test_indices, :, :]
        labels_train = labels_[train_indices, :, :]
        labels_train_selected = labels_train[:, 0, :]
        labels_test_selected = labels_[test_indices, :, :][:, 0, :]

        # stage 1
        bootstrap_Ws = []
        for _ in tqdm(range(n_bootstraps), desc="Bootstraps"):
            bootstrap_indices = torch.randint(0, N_train, (N_train,), device=f'cuda:{gpuid}')
            boot_inputs_train = inputs_train[bootstrap_indices, :, :]
            boot_labels_train = labels_train[bootstrap_indices, :, :]
            
            auc_test_CVs = []
            for _ in tqdm(range(n_CV), desc="Cross Validation"):
                CV_indices = torch.randperm(N_train)
                CV_train_indices = CV_indices[:N_CV_train]
                CV_test_indices = CV_indices[N_CV_train:]
                CV_inputs_train = boot_inputs_train[CV_train_indices, :, :]
                CV_labels_train = boot_labels_train[CV_train_indices, :, :]
                CV_inputs_test = boot_inputs_train[CV_test_indices, :, :]
                CV_labels_test = boot_labels_train[CV_test_indices, :, :]
                
                W = torch.randn((P, n_rhos, B), device=f'cuda:{gpuid}', requires_grad=False)
                pbar = tqdm(range(n_epochs), desc="Sparse Training")
                for epoch in pbar:
                    if epoch > 0:
                        prev_loss = loss.detach().clone()
                        prev_W = W.detach().clone()
                    
                    logits = torch.einsum('ijp,jkp->ikp', CV_inputs_train, W) # shape: (N_train, n_rhos, B)
                    probs = torch.sigmoid(logits)
                    error = probs - CV_labels_train  # shape: (N_train, n_rhos, B)
                    grad = torch.einsum('ijp,ikp->jkp', CV_inputs_train, error) / N_CV_train # shape: (P, n_rhos, B)
                    W = soft_thresholding(W - lr * grad, lr * rho_seqs_)

                    loss = -Bernoulli(probs=probs).log_prob(CV_labels_train).mean()
                    reg_term = (rho_seqs_ * torch.norm(W, p=1, dim=0)).mean()
                    if epoch > 0:
                        d_loss = (prev_loss - loss).item()
                        d_W = torch.norm(prev_W - W, p=2).item()
                        grad_norm = torch.norm(grad, p=2).item()
                        pbar.set_postfix({
                            "loss": loss.item(),
                            "reg_term": reg_term.item(),
                            "d_loss": d_loss,
                            "d_W": d_W,
                            "grad_norm": grad_norm,
                        })
                        if d_loss < eps and d_W < eps and grad_norm < eps:
                            break
                
                logits_test = torch.einsum('ijp,jkp->ikp', CV_inputs_test, W) # shape: (N_test, n_rhos, B)
                auc_test_Bs = []
                for j in tqdm(range(B), desc="Calculate AUC"):
                    auc_test_rhos = []
                    for k in range(n_rhos):
                        auc_test_rho = auroc(
                            torch.sigmoid(logits_test[:,k,j].detach().cpu()),
                            CV_labels_test[:,k,j].detach().cpu(),
                        )
                        auc_test_rhos.append(auc_test_rho)
                        auroc.reset()
                    auc_test_rhos = torch.stack(auc_test_rhos, dim=0) # shape: (n_rhos,)
                    auc_test_Bs.append(auc_test_rhos)
                auc_test_Bs = torch.stack(auc_test_Bs, dim=1) # shape: (n_rhos, B)
                auc_test_CVs.append(auc_test_Bs)
                
            auc_test_CVs = torch.stack(auc_test_CVs, dim=2) # shape: (n_rhos, B, n_seeds)
            auc_test_CVs = auc_test_CVs.mean(dim=2) # shape: (n_rhos, B)
            # pick the largest (first) rho that has highest test AUC
            best_rho_indices = auc_test_CVs.argmax(dim=0) # shape: (B,)
            W = W[:, best_rho_indices, torch.arange(B)]  # shape: (P, B)
            W_binary = (W != 0).float()
            bootstrap_Ws.append(W_binary)
            
        bootstrap_Ws = torch.stack(bootstrap_Ws, dim=2) # shape: (P, B, n_bootstraps)
        bootstrap_Ws = torch.mean(bootstrap_Ws, dim=2) # shape: (P, B)
        torch.save(bootstrap_Ws, f'{output_dir}/bootstrap_W_{i}.pt')

        # stage 2
        selected_idx = (bootstrap_Ws > freq_threshold) # shape: (P, B)
        selected_idx = selected_idx.unsqueeze(0).expand(N_train, -1, -1) # shape: (N, P, B)
        inputs_train_selected = inputs_train * selected_idx # shape: (N, P, B)
        W = torch.randn((P, B), device=f'cuda:{gpuid}', requires_grad=True)
        optimizer = torch.optim.LBFGS([W], lr=0.01, max_iter=1000, history_size=10, line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            logits = torch.einsum('ijp,jp->ip', inputs_train_selected, W) # shape: (N_train, B)
            probs = torch.sigmoid(logits)
            loss = -Bernoulli(probs=probs).log_prob(labels_train_selected).mean()
            loss.backward()
            return loss
        
        W = trainer([W], optimizer, closure, n_epochs_refit)[0] # shape: (P, B)
        Ws.append(W.detach())
        torch.save(W, f'{output_dir}/W_{i}.pt')

        logits_train = torch.einsum('ijp,jp->ip', inputs_train, W) # shape: (N, B)
        final_auc_train = auroc(torch.sigmoid(logits_train), labels_train_selected)
        logits_test = torch.einsum('ijp,jp->ip', inputs_test, W)
        final_auc_test = auroc(torch.sigmoid(logits_test), labels_test_selected)
        print(f"final_auc_train: {final_auc_train}; final_auc_test: {final_auc_test}")
        f.write(f"final_auc_train: {final_auc_train}; final_auc_test: {final_auc_test}\n")

    Ws = torch.cat(Ws, dim=1) # shape: (P, P)
    non_zero_mask = (Ws != 0) & (Ws.T != 0)
    Ws_avg = 0.5 * (Ws + Ws.T)
    Ws = torch.where(non_zero_mask, Ws_avg, torch.zeros_like(Ws))
    Ws[torch.arange(Ws.size(0)), torch.arange(Ws.size(1))] = 0
    torch.save(Ws, f'{output_dir}/W.pt')
    W_df = pd.DataFrame(Ws.cpu().numpy(), columns=input_df.columns)
    W_df.to_csv(f'{output_dir}/W.csv', index=False)

    non_zero_count = (Ws != 0).sum().item()
    total_elements = Ws.numel()
    percent_non_zero = 100.0 * non_zero_count / total_elements
    print(f"Non-zero elements: {non_zero_count} / {total_elements} ({percent_non_zero:.2f}%)")
    f.write(f"Non-zero elements: {non_zero_count} / {total_elements} ({percent_non_zero:.2f}%)\n")

    cmap = mcolors.ListedColormap(['blue', 'white', 'red'])
    bounds = [-abs(Ws).max().item(), 0, abs(Ws).max().item()]  # Set the bounds for the color scale
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(8, 6))
    plt.imshow(Ws.cpu().numpy(), cmap=cmap, norm=norm)
    plt.title("Weight Matrix")
    plt.colorbar(label="Weight")
    plt.savefig(f"{output_dir}/weight_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()