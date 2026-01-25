import argparse
from pathlib import Path
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tueplots import bundles
import sys
bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))
from utils import compute_pass_datk_gts, compute_pass_datk_irt
from scipy.special import expit

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "5_generalization_law_curve"
PROB_THRESHOLD = 0.005

parser = argparse.ArgumentParser()
parser.add_argument("--input-hf-repo", type=str, default="irsl_testtime_resmat2", choices=["irsl_testtime_resmat1", "irsl_testtime_resmat2"])
args = parser.parse_args()

est_path = DATA_DIR / f"4_generalization_estimation_{args.input_hf_repo}.pt"
est_payload = torch.load(est_path, map_location="cpu", weights_only=False)

for dataset in tqdm(sorted(est_payload.keys()), desc="datasets"):
    bench_info = est_payload[dataset]
    test_models = list(bench_info["model_names"])
    theta_from_easy = np.array(bench_info["theta_from_easy"], dtype=np.float32)
    easy_zs = np.array(bench_info["easy_zs"], dtype=np.float32)
    hard_zs = np.array(bench_info["hard_zs"], dtype=np.float32)
    easy_tensor = np.array(bench_info["easy_tensor"], dtype=np.float32)
    hard_tensor = np.array(bench_info["hard_tensor"], dtype=np.float32)

    output_dir = RESULTS_DIR / args.input_hf_repo / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = easy_tensor.shape[-1]
    sample_arange = np.arange(1, n_samples + 1)
    
    for model_idx, model_name in enumerate(test_models):
        easy_model_tensor = easy_tensor[model_idx]
        hard_model_tensor = hard_tensor[model_idx]

        easy_mask = np.nanmean(easy_model_tensor, axis=-1) >= PROB_THRESHOLD
        hard_mask = np.nanmean(hard_model_tensor, axis=-1) >= PROB_THRESHOLD
        easy_model_tensor = easy_model_tensor[easy_mask]
        hard_model_tensor = hard_model_tensor[hard_mask]
        easy_zs_masked = easy_zs[easy_mask]
        hard_zs_masked = hard_zs[hard_mask]

        pass_easy_gt = compute_pass_datk_gts(easy_model_tensor)
        pass_hard_gt = compute_pass_datk_gts(hard_model_tensor)

        theta = float(theta_from_easy[model_idx])
        easy_probs = expit(theta + easy_zs_masked)
        hard_probs = expit(theta + hard_zs_masked)
        pass_easy_est = compute_pass_datk_irt(easy_probs, n_samples)
        pass_hard_est = compute_pass_datk_irt(hard_probs, n_samples)

        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.loglog(sample_arange, -np.log(pass_easy_gt), label="Easy GT", color="blue")
            ax.loglog(sample_arange, -np.log(pass_hard_gt), label="Hard GT", color="red")
            ax.loglog(sample_arange, -np.log(pass_easy_est), label="Easy Est", color="blue", linestyle="--")
            ax.loglog(sample_arange, -np.log(pass_hard_est), label="Hard Est", color="red", linestyle="--")
            ax.set_xlabel("Number of Samples", fontsize=16)
            ax.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
            ax.legend(fontsize=12)
            ax.tick_params(axis="both", labelsize=12)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"hardeasy_{args.input_hf_repo.replace('4_generalization_estimation_irsl_testtime_', '')}_{model_name}_{dataset}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
