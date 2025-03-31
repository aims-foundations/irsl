import pickle
import torch
torch.manual_seed(0)
from tqdm import tqdm
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles

def estimate_theta(asked_ys, asked_zs, theta=0):
    def closure():
        optim.zero_grad()
        probs = torch.sigmoid(theta[:, None] + asked_zs[None, :])
        loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        loss.backward()
        return loss

    asked_ys = torch.tensor(asked_ys)
    asked_zs = torch.tensor(asked_zs)
    theta = theta.clone().requires_grad_(True)
    optim = torch.optim.LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
    for iteration in range(100):
        if iteration > 0:
            previous_theta = theta.clone()
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = previous_loss - loss
            d_theta = torch.norm(previous_theta - theta, p=2)
            grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
            if d_loss < 1e-5 and d_theta < 1e-5 and grad_norm < 1e-5:
                break
    
    return theta.detach()

if __name__ == "__main__":
    device = "cuda:4"
    
    # Load the results DataFrame which has a MultiIndex with levels "request.prompt", "z", "scenario"
    with open("../data/results.pkl", "rb") as f:
        results = pickle.load(f)
    
    mask = ~results.columns.get_level_values("z").isna()
    results = results.loc[:, mask]

    # Convert the DataFrame values into a torch tensor for the response matrix.
    data = torch.tensor(results.values, dtype=torch.float, device=device)
    n_test_takers, n_items = data.shape
    
    # Extract the z-values from the DataFrame columns (from level "z").
    z_values = results.columns.get_level_values("z").astype(float).to_numpy()
    z_tensor = torch.tensor(z_values, dtype=torch.float, device=device)
    
    # Estimate theta for each test taker (each row in the response matrix)
    estimated_thetas = []
    for i in tqdm(range(n_test_takers), desc="Estimating theta for each test taker"):
        asked_ys = data[i, :]  # responses for test taker i
        # Initialize theta as a tensor (starting point 0) and estimate theta.
        theta_est = estimate_theta(asked_ys, z_tensor, theta=torch.tensor([0.0], device=device))
        estimated_thetas.append(theta_est.item())
    
    # Assuming the row index names contain the time step information,
    # e.g., "step1", "step2", etc. We extract the numeric part:
    time_steps = [int(name.split("step")[-1]) for name in results.index]
    
    # Plot the estimated theta values versus time steps using the ICML2024 TeX-style context.
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(8, 6))
        plt.plot(time_steps, estimated_thetas, marker="o", linestyle="-")
        plt.xlabel("Time Step")
        plt.ylabel("Estimated Theta")
        plt.title("Estimated Theta vs. Time Step")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("theta_vs_time.png", dpi=300)
        plt.show()
    
    print("Estimated theta values for each test taker:")
    print(estimated_thetas)
