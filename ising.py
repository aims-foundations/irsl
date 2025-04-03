import pandas as pd
import torch
from tqdm import tqdm
from torchmetrics import AUROC
import torch.nn.functional as F
from torch.distributions import Bernoulli
BCE = F.binary_cross_entropy
auroc = AUROC(task="binary")
verbose = True
gpuid = 0

data = torch.tensor(pd.read_csv('MetData.csv').values, dtype=torch.float32)
N, P = data.shape
subsets = []
leftovers = []
n_rhos = 100

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
rho_seqs = torch.stack(rho_seqs, dim=1) # shape: (n_rhos, P)

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

step_size = 1024
for i in tqdm(range(0, P, step_size)):
    B = min(step_size, P - i)
    W = torch.randn((P, n_rhos, B), device=f'cuda:{gpuid}', requires_grad=True)
    optimizer = torch.optim.LBFGS([W], lr=0.01, max_iter=1000, history_size=10, line_search_fn='strong_wolfe')

    inputs_ = inputs[:, :, i:i+B].to(f'cuda:{gpuid}')
    labels_ = labels[:, :, i:i+B].to(f'cuda:{gpuid}')
    rho_seqs_ = rho_seqs[:, i:i+B].to(f'cuda:{gpuid}')

    def closure():
        optimizer.zero_grad()
        logits = torch.einsum('ijp,jkp->ikp', inputs_, W) # shape: (N, n_rhos, P)
        probs = torch.sigmoid(logits)
        loss = -Bernoulli(probs=probs).log_prob(labels_) # shape: (N, n_rhos, P)
        loss = loss.mean(dim=0) + rho_seqs_ * torch.norm(W, p=1, dim=0) # >> shape: (n_rhos, P)
        loss = loss.mean()
        loss.backward()
        return loss

    W = trainer([W], optimizer, closure, verbose=True)[0]
    logits = torch.einsum('ijp,jkp->ikp', inputs_, W) # shape: (N, n_rhos, P)
    auc = auroc(torch.sigmoid(logits), labels_)
    print(f"AUC: {auc.item()}")