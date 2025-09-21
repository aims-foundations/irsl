import numpy as np
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from tqdm import tqdm
import sys
sys.path.append("../..")
from tueplots import bundles
bundles.icml2024()
from huggingface_hub import snapshot_download
torch.manual_seed(0)
from torch.optim import LBFGS
from torch.distributions import Bernoulli
from scipy.stats import spearmanr

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
    B = 50000
    n_thetas_nuisance = 150
    device = "cuda:7"
    
    cache_dir = snapshot_download(repo_id="stair-lab/irsl_testtime_resmat2", repo_type="dataset")
    testtime_resmat2 = torch.load(f"{cache_dir}/resmat.pt", map_location="cpu", weights_only=False)
    data_tensor = testtime_resmat2["data_tensor"]
    print(data_tensor.shape)
    model_names = testtime_resmat2["models"]
    datasets  = testtime_resmat2["datasets"]
    questions   = testtime_resmat2["questions"]
    
    resmat = data_tensor[:15, :, 0]
    print(resmat.shape)
    nan_count = torch.isnan(resmat).sum().item()
    total_count = resmat.numel()
    nan_percentage = nan_count / total_count * 100
    print(f"nan count: {nan_count}, nan percentage: {nan_percentage:.2f}%")

    data = torch.tensor(resmat, dtype=torch.float, device=device)
    n_test_takers, n_items = data.shape
    optimized_z = []
    thetas_nuisance = torch.randn(n_thetas_nuisance, n_test_takers, device=device)
    for i in tqdm(range(0, n_items, B)):
        data_batch = data[:, i:i+B]
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

    rho1, _ = spearmanr(zs.cpu().numpy(), resmat.mean(0))
    rho2, _ = spearmanr(zs.cpu().numpy(), torch.nanmean(data_tensor[:15, :, :], dim=(0, 2)))
    rho3, _ = spearmanr(zs.cpu().numpy(), torch.nanmean(data_tensor[15:, :, :], dim=(0, 2)))
    print(f"\
        corr with resmat.mean(0) = {rho1:.6f}, \
        \ncorr with torch.nanmean(data_tensor[:15, :, :], dim=(0, 2)) = {rho2:.6f} \
        \ncorr with torch.nanmean(data_tensor[15:, :, :], dim=(0, 2)) = {rho3:.6f} \
    ")
    
    last4_data_tensor = data_tensor[-4:]
    last4_model_names = model_names[-4:]
    zs_list = zs.cpu().numpy().tolist()
    assert len(questions) == len(datasets) == len(zs_list) == last4_data_tensor.shape[1]
    assert len(last4_model_names) == last4_data_tensor.shape[0]
    print(last4_data_tensor.shape)
    out_path = "irsl_testtime_resmat2_withz.pt"
    payload = {
        "data_tensor": last4_data_tensor,
        "models": last4_model_names,
        "questions": questions,
        "datasets": datasets,
        "zs": zs_list,
    }
    torch.save(payload, out_path)