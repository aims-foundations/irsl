from pathlib import Path
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from tueplots import bundles
from scipy.special import expit
import sys
bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))
from utils import calibrate_1pl_theta, compute_pass_datk_gts, compute_pass_datk_irt

DEVICE = "cuda:7"
PROB_THRESHOLD = 0.005

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "6_generalization_across_aime"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

input_path = DATA_DIR / "1_calibreated_irsl_testtime_resmat2.pt"
payload = torch.load(input_path, map_location="cpu", weights_only=False)

data_tensor = np.array(payload["data_tensor"], dtype=np.float32)
model_names = list(payload["models"])
test_models = list(payload["test_models"])
test_model_indices = [i for i, m in enumerate(model_names) if m in set(test_models)]
datasets = list(payload["datasets"])
zs = np.array(payload["zs"], dtype=np.float32)

aime24_indices = [i for i, d in enumerate(datasets) if d == "aime2024"]
aime25_indices = [i for i, d in enumerate(datasets) if d == "aime2025"]
bench_tensor_24 = data_tensor[test_model_indices][:, aime24_indices, :]
bench_tensor_25 = data_tensor[test_model_indices][:, aime25_indices, :]

probmat_24 = np.nanmean(bench_tensor_24, axis=-1)
zs_24 = torch.tensor(zs[aime24_indices])
thetas = calibrate_1pl_theta(torch.tensor(probmat_24), DEVICE, zs_24)

n_samples = data_tensor.shape[-1]
sample_arange = np.arange(1, n_samples + 1)
for model_idx, model_name in enumerate(test_models):
    model_tensor_24 = bench_tensor_24[model_idx]
    model_tensor_25 = bench_tensor_25[model_idx]

    mask_24 = np.nanmean(model_tensor_24, axis=-1) >= PROB_THRESHOLD
    mask_25 = np.nanmean(model_tensor_25, axis=-1) >= PROB_THRESHOLD
    model_tensor_24 = model_tensor_24[mask_24]
    model_tensor_25 = model_tensor_25[mask_25]
    zs_24_masked = zs[aime24_indices][mask_24]
    zs_25_masked = zs[aime25_indices][mask_25]

    pass_24_gt = compute_pass_datk_gts(model_tensor_24)
    pass_25_gt = compute_pass_datk_gts(model_tensor_25)

    theta = float(thetas[model_idx])
    probs_24 = expit(theta + zs_24_masked)
    probs_25 = expit(theta + zs_25_masked)
    pass_24_est = compute_pass_datk_irt(probs_24, n_samples)
    pass_25_est = compute_pass_datk_irt(probs_25, n_samples)

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.loglog(sample_arange, -np.log(pass_24_gt), label="AIME24 GT", color="blue")
        ax.loglog(sample_arange, -np.log(pass_24_est), label="AIME24 Est", color="blue", linestyle="--")
        ax.loglog(sample_arange, -np.log(pass_25_gt), label="AIME25 GT", color="red")
        ax.loglog(sample_arange, -np.log(pass_25_est), label="AIME25 Est", color="red", linestyle="--")
        ax.set_title(model_name, fontsize=16)
        ax.set_xlabel("Number of Samples", fontsize=16)
        ax.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"generalization_across_aime_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
