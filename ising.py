import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.distributions import Bernoulli
from torchmetrics import AUROC
auroc = AUROC(task="binary")
verbose = True
gpuid = 1

data = torch.tensor(pd.read_csv('gsm_hard_easy_10.csv').values, dtype=torch.float32)
N, P = data.shape
subsets = []
leftovers = []
n_rhos = 100
gamma = 0.25
train_percentage = 0.86
N_train = int(N*train_percentage)

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
labels = torch.cat(leftovers, dim=2) # shape: (N, 1, P)
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

def trainer(parameters, optim, closure, verbose=True, epochs=1000):
    pbar = tqdm(range(epochs)) if verbose else range(epochs)
    loss = closure()

    for iteration in pbar:
        previous_parameters = [p.clone().detach() for p in parameters]
        previous_loss = loss.detach()

        loss = optim.step(closure)

        d_loss = (previous_loss - loss).item()
        d_parameters = sum(
            torch.norm(prev - curr, p=2).item()
            for prev, curr in zip(previous_parameters, parameters)
        )
        grad_norm = torch.norm(W.grad, p=2).item()
        if verbose:
            pbar.set_postfix({
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "d_parameter": d_parameters,
                "d_loss": d_loss
            })
        eps = 1e-5
        if d_loss < eps and d_parameters < eps and grad_norm < eps:
            break

    return parameters

auc_trains = []
auc_tests = []
for seed in range(1):
    torch.manual_seed(seed)
    step_size = 1024
    Ws = []
    picked_rhos = []
    for i in tqdm(range(0, P, step_size)):
        B = min(step_size, P - i)
        W = torch.randn((P, n_rhos, B), device=f'cuda:{gpuid}', requires_grad=True)
        optimizer = torch.optim.LBFGS([W], lr=0.01, max_iter=1000, history_size=10, line_search_fn='strong_wolfe')

        indices = torch.randperm(N)
        train_indices = indices[:N_train]
        test_indices = indices[N_train:]
        inputs_ = inputs[:, :, i:i+B].to(f'cuda:{gpuid}') # shape: (N, P-1, B)
        inputs_train = inputs_[train_indices, :, :]
        inputs_test = inputs_[test_indices, :, :]
        labels_ = labels[:, :, i:i+B].to(f'cuda:{gpuid}') # shape: (N, n_rhos, B)
        labels_train = labels_[train_indices, :, :]
        labels_test = labels_[test_indices, :, :]
        rho_seqs_ = rho_seqs[:, i:i+B].to(f'cuda:{gpuid}') # shape: (n_rhos, B)

        def closure():
            optimizer.zero_grad()
            logits = torch.einsum('ijp,jkp->ikp', inputs_train, W) # shape: (N_train, n_rhos, B)
            probs = torch.sigmoid(logits)
            loss = -Bernoulli(probs=probs).log_prob(labels_train) # shape: (N_train, n_rhos, B)
            loss = loss.mean(dim=0) + rho_seqs_ * torch.norm(W, p=1, dim=0) # shape: (n_rhos, B)
            loss = loss.mean()
            loss.backward()
            return loss

        W = trainer([W], optimizer, closure, verbose=True)[0] # shape: (P, n_rhos, B)
        logits_test = torch.einsum('ijp,jkp->ikp', inputs_test, W) # shape: (N_test, n_rhos, B)
        best_rho_indices = []
        for j in range (B):
            auc_candidates = []
            for i in range (n_rhos):
                auc_candidate = auroc(torch.sigmoid(logits_test[:,i,j]), labels_test[:,i,j])
                auc_candidates.append(auc_candidate)
            auc_candidates = torch.stack(auc_candidates, dim=0) # shape: (n_rhos,)
            best_rho_idx = auc_candidates.argmax(dim=0)
            best_rho_indices.append(best_rho_idx)
        best_rho_indices = torch.stack(best_rho_indices, dim=0) # shape: (B,)
        picked_rho = rho_seqs_[best_rho_indices, torch.arange(B)]  # shape: (B,)
        picked_rhos.append(picked_rho.cpu())
        W = W[:, best_rho_indices, torch.arange(B)]  # shape: (P, B)
        
        logits_train = torch.einsum('ijp,jp->ip', inputs_train, W) # shape: (N, B)
        auc_train = auroc(torch.sigmoid(logits_train), labels_train[:,0,:])
        auc_trains.append(auc_train.item())
        logits_test = torch.einsum('ijp,jp->ip', inputs_test, W)
        auc_test = auroc(torch.sigmoid(logits_test), labels_test[:,0,:])
        auc_tests.append(auc_test.item())
        Ws.append(W)
        torch.save(W, f'W_{i}.pt')

    Ws = torch.cat(Ws, dim=1) # shape: (P, P)
    torch.save(Ws, 'W.pt')

    picked_rhos = torch.cat(picked_rhos, dim=0)  # shape: (P,)
    df_rhos = pd.DataFrame({
        "min_rhos": min_rhos.numpy(),
        "picked_rho": picked_rhos.numpy(),
        "max_rhos": max_rhos.numpy(),
    })
    df_rhos.to_csv("final_picked_rho.csv", index=False)

print(f"Train AUC: {np.mean(auc_trains):.4f} ± {np.std(auc_trains):.4f}")
print(f"Test AUC:  {np.mean(auc_tests):.4f} ± {np.std(auc_tests):.4f}")
