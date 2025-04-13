import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils import resample
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from torch.distributions import Bernoulli
import torch
from torchmetrics import AUROC
auroc = AUROC(task="binary")
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles
bundles.icml2024()

def binary_neighborhood_selection(X, max_workers=4):
    """
    Stability selection with bootstrapped sparse logistic regression (LogisticRegressionCV).

    Args:
        X: (N, P) binary matrix (entries should be 0 or 1).

    Returns:
        edge_freq: (P, P) matrix of edge selection frequencies.
    """
    N, P = X.shape
    n_bootstraps = 100

    def process_bootstrap(b):
        X_b = resample(X, replace=True, n_samples=N)
        local_edge_counts = np.zeros((P, P))

        def process_feature(j):
            y = X_b[:, j]
            X_others = np.delete(X_b, j, axis=1)
            feature_indices = [i for i in range(P) if i != j]

            # Logistic regression with L1 regularization and cross-validation
            model = LogisticRegressionCV(
                penalty='l1',
                solver='saga',
                cv=10,
                scoring='neg_log_loss',
                max_iter=10000,
                fit_intercept=False,
                Cs=10,  # number of inverse regularization values
                tol=1e-4,
            )

            try:
                model.fit(X_others, y)
                coefs = model.coef_[0] # shape (P - 1,)
            except Exception as e:
                # Catch numerical issues (e.g. perfect separation) and skip
                coefs = np.zeros(P - 1)

            # Collect the feature indices for which the coefficient is nonzero.
            selected_edges = []
            for idx, coef in zip(feature_indices, coefs):
                if coef != 0:
                    selected_edges.append((j, idx))
            return selected_edges

        with ThreadPoolExecutor() as executor_inner:
            results = list(executor_inner.map(process_feature, range(P)))

        for selected_edges in results:
            for (j, idx) in selected_edges:
                local_edge_counts[j, idx] += 1

        return local_edge_counts

    total_edge_counts = np.zeros((P, P))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_bootstrap, b) for b in range(n_bootstraps)]
        for future in tqdm(as_completed(futures), total=n_bootstraps, desc="Bootstraps"):
            local_edge_counts = future.result()
            total_edge_counts += local_edge_counts

    edge_freq = total_edge_counts / n_bootstraps
    edge_freq = np.maximum(edge_freq, edge_freq.T)

    return edge_freq

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
        grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
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

if __name__ == "__main__":
    gpuid = 4
    n_epochs_refit = 10000
    train_percentage = 0.86
    
    file_name = 'gsm_hard_easy_200.csv'
    output_dir = f"result/ising/{file_name.split('.')[0]}_sklearn"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(file_name)
    X = df.values
    N, P = X.shape
    print(X.shape)
    
    N_train = int(N * train_percentage)
    indices = np.random.permutation(N)
    train_indices = indices[:N_train]
    test_indices = indices[N_train:]
    
    # stage 1: variable selection
    X_train = X[train_indices, :]
    edge_freq = binary_neighborhood_selection(X_train)
    with open(f"{output_dir}/edge_freq.pkl", 'wb') as f:
        pickle.dump(edge_freq, f)
    
    selected_idx = (edge_freq >= 0.90).astype(float) # shape: (P, P)
    selected_idx_df = pd.DataFrame(selected_idx, columns=df.columns)
    selected_idx_df.to_csv(f"{output_dir}/selected_idx.csv", index=False)
        
    # stage 2: refitting logistic regression
    data = torch.tensor(X, dtype=torch.float32)
    subsets = []
    leftovers = []
    for i in range(P):
        # Mask to remove i-th column
        mask = torch.ones(P, dtype=torch.bool)
        mask[i] = False
        subset = data[:, mask] # shape: (N, P-1)
        # add a one column to the subset
        subset = torch.cat((subset, torch.ones(N, 1)), dim=1) # shape: (N, P)
        leftover = data[:, i] # shape: (N,)
        subsets.append(subset.unsqueeze(2)) # shape: (N, P-1, 1)
        leftovers.append(leftover.unsqueeze(1)) # shape: (N, 1)
    inputs = torch.cat(subsets, dim=2).to(f'cuda:{gpuid}') # shape: (N, P-1, P)
    labels = torch.cat(leftovers, dim=2).to(f'cuda:{gpuid}') # shape: (N, P)
    inputs_train = inputs[train_indices, :, :]
    inputs_test = inputs[test_indices, :, :]
    labels_train = labels[train_indices, :]
    labels_test = labels[test_indices, :]
    
    selected_idx_N_train = torch.tensor(selected_idx).unsqueeze(0).expand(N_train, -1, -1) # shape: (N_train, P, P)
    inputs_train_selected = inputs * selected_idx_N_train # shape: (N_train, P)
    W = torch.randn((P, P), device=f'cuda:{gpuid}', requires_grad=True)
    optimizer = torch.optim.LBFGS([W], lr=0.01, max_iter=1000, history_size=10, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        logits = torch.einsum('ijp,jp->ip', inputs_train_selected, W) # shape: (N_train, B)
        probs = torch.sigmoid(logits)
        loss = -Bernoulli(probs=probs).log_prob(labels_train).mean()
        loss.backward()
        return loss
    
    W = trainer([W], optimizer, closure, n_epochs_refit)[0] # shape: (P, B)
    W =  W * selected_idx
    torch.save(W, f'{output_dir}/W.pt')
    logits_train = torch.einsum('ijp,jp->ip', inputs_train, W) # shape: (N, B)
    final_auc_train = auroc(torch.sigmoid(logits_train), labels_train)
    logits_test = torch.einsum('ijp,jp->ip', inputs_test, W)
    final_auc_test = auroc(torch.sigmoid(logits_test), labels_test)
    print(f"final_auc_train: {final_auc_train}; final_auc_test: {final_auc_test}")
    f.write(f"final_auc_train: {final_auc_train}; final_auc_test: {final_auc_test}\n")
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.subplot(1, 3, 1)
        plt.imshow(edge_freq, cmap='Blues')
        plt.title("Edge Frequencies")
        plt.colorbar(label="Frequency")

        plt.subplot(1, 3, 2)
        plt.imshow(selected_idx, cmap='Greys')
        plt.title("Recovered Graph")

        plt.subplot(1, 3, 3)
        cmap = mcolors.ListedColormap(['blue', 'white', 'red'])
        bounds = [-abs(W).max().item(), 0, abs(W).max().item()]  # Set the bounds for the color scale
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(W.cpu().numpy(), cmap=cmap, norm=norm)
        plt.title("Weight Matrix")
        plt.colorbar(label="Weight", ticks=[-abs(W).max().item(), 0, abs(W).max().item()])
        plt.savefig(f"{output_dir}/weight_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/glasso.png", dpi=300, bbox_inches="tight")