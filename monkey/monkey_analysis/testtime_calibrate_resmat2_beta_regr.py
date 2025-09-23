import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from tqdm import tqdm
import sys
sys.path.append("../..")
from utils import beta_nll, trainer
from tueplots import bundles
bundles.icml2024()
from huggingface_hub import snapshot_download
torch.manual_seed(0)
from torch.optim import LBFGS
from scipy.stats import spearmanr
    
if __name__ == "__main__":
    B = 50000
    n_thetas_nuisance = 150
    device = "cuda:7"
    N_MODELS_FOR_PROP = 8
    N_SAMPLES_FOR_PROP = 100
    eps = 1e-6

    cache_dir = snapshot_download(repo_id="stair-lab/irsl_testtime_resmat2", repo_type="dataset")
    testtime_resmat2 = torch.load(f"{cache_dir}/resmat.pt", map_location="cpu", weights_only=False)
    data_tensor = testtime_resmat2["data_tensor"]
    model_names = testtime_resmat2["models"]
    datasets    = testtime_resmat2["datasets"]
    questions   = testtime_resmat2["questions"]
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
    print(last4_data_tensor.shape)
    breakpoint()
    out_path = "irsl_testtime_resmat2_withz_betareg.pt"
    payload = {
        # "data_tensor": last4_data_tensor,
        "data_tensor": data_tensor,
        # "models": last4_model_names,
        "models": model_names,
        "questions": questions,
        "datasets": datasets,
        "zs": zs_list,
    }
    torch.save(payload, out_path)