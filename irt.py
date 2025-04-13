from tqdm import tqdm
from torchmetrics import AUROC
auroc = AUROC(task="binary")
import torch
from torch.distributions import Bernoulli
from torch.optim import LBFGS
import pandas as pd
import os

def trainer(parameters, optim, closure, n_iter=100, verbose=True):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for iteration in pbar:
        if iteration > 0:
            # Clone each tensor individually for previous state
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss})
            
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                break
            
    return parameters

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda:0"
    B = 50000
    n_thetas_nuisance = 150
    test_takers_train_percentage = 0.8
    items_train_percentage = 0.5
    
    file_name = 'resmat_lite_all.csv'
    output_dir = f"result/ising/{file_name.split('.')[0]}_irt"
    os.makedirs(output_dir, exist_ok=True)
    input_df = pd.read_csv(file_name)
    data = torch.tensor(input_df.values, dtype=torch.float32, device=device)
    n_test_takers, n_items = data.shape
    print(data.shape)
    
    n_test_takers_train = int(n_test_takers*test_takers_train_percentage)
    test_takers_indices = torch.randperm(n_test_takers)
    test_takers_train_indices = test_takers_indices[:n_test_takers_train]
    test_takers_test_indices = test_takers_indices[n_test_takers_train:]
    
    n_items_train = int(n_items*items_train_percentage)
    items_indices = torch.randperm(n_items)
    items_train_indices = items_indices[:n_items_train]
    items_test_indices = items_indices[n_items_train:]
    
    data_train_z = data[test_takers_train_indices, :]
    data_train_theta = data[test_takers_test_indices, :][:, items_train_indices]
    data_test = data[test_takers_test_indices, :][:, items_test_indices]
    
    optimized_z = []
    thetas_nuisance = torch.randn(n_thetas_nuisance, n_test_takers_train, device=device)
    for i in tqdm(range(0, n_items, B)):
        data_batch = data_train_z[:, i:i+B]
        current_B = data_batch.shape[1]
        z = torch.randn(current_B, requires_grad=True, device=device)
        optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        def closure_z():
            optim_z.zero_grad()
            mask = ~torch.isnan(data_batch).expand(n_thetas_nuisance, -1, -1)
            probs = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
            loss = -(Bernoulli(probs=probs[mask]).log_prob(
                data_batch.expand(n_thetas_nuisance, -1, -1)[mask]
            )).mean()
            loss.backward()
            return loss
        z_optimized = trainer([z], optim_z, closure_z)[0].detach()
        optimized_z.append(z_optimized)
    zs = torch.cat(optimized_z)
    
    thetas = torch.randn(n_test_takers - n_test_takers_train, requires_grad=True, device=device)
    optim_theta = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_theta():
        optim_theta.zero_grad()
        mask = ~torch.isnan(data_train_theta)
        probs = torch.sigmoid(thetas[:, None] + zs[items_train_indices][None, :])
        loss = -(Bernoulli(probs=probs[mask]).log_prob(data_train_theta[mask])).mean()
        loss.backward()
        return loss
    thetas = trainer([thetas], optim_theta, closure_theta)[0].detach()
    
    mask = ~torch.isnan(data_test)
    probs = torch.sigmoid(thetas[:, None] + zs[items_test_indices][None, :])
    auc_test = auroc(probs[mask], data_test[mask])
    with open(f"{output_dir}/print.txt", "a") as f:
        f.write(f"auc_test: {auc_test.item()}\n")
    print(f"auc_test: {auc_test.item()}")