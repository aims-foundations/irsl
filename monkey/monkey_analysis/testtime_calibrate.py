import numpy as np
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from torch.optim import LBFGS
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download
import sys
sys.path.append("../..")
from utils import beta_nll, trainer
from tueplots import bundles
bundles.icml2024()
    
if __name__ == "__main__":
    eps = 1e-6
    n_thetas_nuisance = 150
    device = "cuda:7"
    N_MODELS_FOR_TEST = 4
    
    # FILE_NAME = "irsl_testtime_resmat1"
    FILE_NAME = "irsl_testtime_resmat2"
    cache_dir = snapshot_download(repo_id=f"stair-lab/{FILE_NAME}", repo_type="dataset")
    
    testtime_resmat = torch.load(f"{cache_dir}/resmat.pt", map_location="cpu", weights_only=False)
    data_tensor = testtime_resmat["data_tensor"].numpy() 
    model_names = testtime_resmat["models"]
    questions   = testtime_resmat["questions"]
    if FILE_NAME == "irsl_testtime_resmat1":
        helm_zs   = np.array(testtime_resmat["zs"])
        datasets    = testtime_resmat["scenarios"]
    elif FILE_NAME == "irsl_testtime_resmat2":
        datasets    = testtime_resmat["datasets"]
    print(data_tensor.shape)
    n_models_for_train = data_tensor.shape[0] - N_MODELS_FOR_TEST
    n_samples_for_train = data_tensor.shape[-1]

    probmat = torch.nanmean(
        torch.tensor(data_tensor[:n_models_for_train, :, :n_samples_for_train], dtype=torch.float, device=device),
        dim = -1,
    )
    n_test_takers, n_items = probmat.shape
    assert not torch.isnan(probmat).any()
    
    thetas_nuisance = torch.randn(n_thetas_nuisance, n_test_takers, device=device)
    z = torch.randn(n_items, requires_grad=True, device=device)
    optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_z():
        optim_z.zero_grad()
        mask = ~torch.isnan(probmat).expand(n_thetas_nuisance, -1, -1)
        mu = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
        y = probmat.expand(n_thetas_nuisance, -1, -1).clamp(eps, 1 - eps)
        phi = torch.tensor(5.0, device=device)
        nll = beta_nll(y[mask], mu[mask], phi)
        loss = nll.mean()
        loss.backward()
        return loss
    z_optimized = trainer([z], optim_z, closure_z, verbose=True)[0].detach().cpu().numpy()

    rho2, _ = spearmanr(z_optimized, np.nanmean(data_tensor[:n_models_for_train, :, :], axis=(0, 2)))
    rho3, _ = spearmanr(z_optimized, np.nanmean(data_tensor[n_models_for_train:, :, :], axis=(0, 2)))
    print(f"\
        \ncorr with torch.nanmean(data_tensor[:n_models_for_train, :, :], dim=(0, 2)) = {rho2:.6f} \
        \ncorr with torch.nanmean(data_tensor[n_models_for_train:, :, :], dim=(0, 2)) = {rho3:.6f} \
    ")
    
    test_model_names = model_names[n_models_for_train:]
    assert len(questions) == len(datasets) == z_optimized.shape[0] == data_tensor.shape[1]
    out_path = f"{FILE_NAME}_withz.pt"
    payload = {
        "data_tensor": data_tensor,
        "models": model_names,
        "test_models": test_model_names,
        "questions": questions,
        "datasets": datasets,
        "zs": z_optimized,
    }
    if FILE_NAME == "irsl_testtime_resmat1":
        payload["helm_zs"] = helm_zs
        
    torch.save(payload, out_path)