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

def beta_nll(y, mu, phi):
    """
    Elementwise negative log-likelihood for Beta(y | a=mu*phi, b=(1-mu)*phi).
    y, mu in (0,1); phi > 0. Broadcasts over inputs.
    """
    a = mu * phi
    b = (1.0 - mu) * phi
    return -((a - 1) * torch.log(y) + (b - 1) * torch.log1p(-y)
             - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)))
    
if __name__ == "__main__":
    B = 50000
    n_thetas_nuisance = 150
    device = "cuda:7"
    N_MODELS_FOR_PROP = 5
    N_SAMPLES_FOR_PROP = 300
    eps = 1e-6

    cache_dir = snapshot_download(repo_id="stair-lab/irsl_testtime_resmat1", repo_type="dataset")
    testtime_resmat1 = torch.load(f"{cache_dir}/testtime_resmat1.pt", map_location="cpu", weights_only=False)
    data_tensor = testtime_resmat1["data_tensor"]
    model_names = testtime_resmat1["models"]
    datasets    = testtime_resmat1["scenarios"]
    questions   = testtime_resmat1["questions"]
    helm_zs   = testtime_resmat1["zs"]
    print("data_tensor shape:", tuple(data_tensor.shape))

    X = torch.mean(
        torch.tensor(data_tensor[:N_MODELS_FOR_PROP, :, :N_SAMPLES_FOR_PROP], dtype=torch.float, device=device),
        dim = -1,
    )
    n_test_takers, n_items = X.shape
    assert not torch.isnan(X).any()
    
    data = torch.tensor(X, dtype=torch.float, device=device)
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
            # model mean in (0,1)
            mu = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
            y = data_batch.expand(n_thetas_nuisance, -1, -1).clamp(eps, 1 - eps)
            # fixed precision (hyperparameter); you can tune this
            phi = torch.tensor(10.0, device=device)  # scalar; broadcasts
            # Beta NLL on valid entries
            nll = beta_nll(y[mask], mu[mask], phi)
            loss = nll.mean()
            loss.backward()
            return loss
        z_optimized = trainer([z], optim_z, closure_z)[0].detach()
        optimized_z.append(z_optimized)
    zs = torch.cat(optimized_z)

    rho2, _ = spearmanr(zs.cpu().numpy(), torch.nanmean(data_tensor[:N_MODELS_FOR_PROP, :, :], dim=(0, 2)))
    rho3, _ = spearmanr(zs.cpu().numpy(), torch.nanmean(data_tensor[N_MODELS_FOR_PROP:, :, :], dim=(0, 2)))
    print(f"\
        \ncorr with torch.nanmean(data_tensor[:N_MODELS_FOR_PROP, :, :], dim=(0, 2)) = {rho2:.6f} \
        \ncorr with torch.nanmean(data_tensor[N_MODELS_FOR_PROP:, :, :], dim=(0, 2)) = {rho3:.6f} \
    ")
    
    last4_data_tensor = data_tensor[N_MODELS_FOR_PROP:]
    last4_model_names = model_names[N_MODELS_FOR_PROP:]
    zs_list = zs.cpu().numpy().tolist()
    assert len(questions) == len(datasets) == len(zs_list) == last4_data_tensor.shape[1]
    assert len(last4_model_names) == last4_data_tensor.shape[0]
    print(last4_model_names)
    out_path = "irsl_testtime_resmat1_withz_betareg.pt"
    breakpoint()
    payload = {
        # "data_tensor": last4_data_tensor,
        "data_tensor": data_tensor,
        # "models": last4_model_names,
        "models": model_names,
        "questions": questions,
        "datasets": datasets,
        "zs": zs_list,
        "helm_zs": helm_zs
    }
    torch.save(payload, out_path)