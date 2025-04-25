import torch
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    repo_id = "EleutherAI/pythia-6.9b_legalbench"
    model_name = repo_id.split("/")[1]
    budget = 30
    
    output_dir = f"result/monitor/{model_name}_budget_{budget}"
    gt_thetas = torch.load(f"{output_dir}/gt_thetas.pt") # shape: (n_test_takers,)
    random_thetass = torch.load(f"{output_dir}/random_thetas.pt")  # shape: (n_test_takers, budget)
    # adaptive_thetass = torch.load(f"{output_dir}/adaptive_thetas.pt")  # shape: (n_test_takers, budget)
    elo_thetass = torch.load(f"{output_dir}/elo_thetas.pt")  # shape: (n_test_takers, budget)
    n_test_takers = gt_thetas.shape[0]

    random_thetas = [random_thetass[i, -1] for i in range(n_test_takers)]
    # adaptive_thetas = [adaptive_thetass[i, -1] for i in range(n_test_takers)]
    elo_thetas = [elo_thetass[i, -1] for i in range(n_test_takers)]
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(6, 6))
        plt.plot(np.arange(len(gt_thetas)), gt_thetas.numpy(), label="Ground Truth", color="black", linewidth=2)
        plt.plot(np.arange(len(random_thetas)), random_thetas, label="Random", color="red", linewidth=2)
        # plt.plot(np.arange(len(adaptive_thetas)), adaptive_thetas, label="Adaptive", color="blue", linewidth=2)
        plt.plot(np.arange(len(elo_thetas)), elo_thetas, label="Elo", color="green", linewidth=2)
        plt.xlabel("Time Step", fontsize=25)
        plt.ylabel("Model Ability", fontsize=25)
        plt.tick_params(axis="both", labelsize=25)
        plt.legend(fontsize=25)
        plt.savefig(f"monitor_ckpt_{model_name}_budget_{budget}.png", dpi=300, bbox_inches="tight")
        plt.close()