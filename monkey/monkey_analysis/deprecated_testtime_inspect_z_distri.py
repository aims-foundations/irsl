import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Define function
def passatk(pass_iat1, k):
    return 1.0 - (1.0 - pass_iat1) ** k

# Parameters
pass_iat1_values = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
k_values = np.arange(1, 101)

# Plot
plt.figure(figsize=(8,6))
for p in pass_iat1_values:
    y = passatk(p, k_values)
    plt.plot(k_values, y, label=f"{p:.3f}")

plt.xlabel("k (number of samples)", fontsize=14)
plt.ylabel("Pass@k", fontsize=14)
plt.title("Pass@k vs k for different pass@1 values", fontsize=16)
plt.legend(title="pass@1")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("passatk_vs_k.png", dpi=300)
plt.close()


def compute_pass_iatk_irt(pass_iat1: float, k: int) -> float:
    return 1.0 - (1.0 - pass_iat1) ** k

def compute_pass_datk_irt(data2d: np.ndarray, irt_probs: np.ndarray) -> np.ndarray:
    n_items, n_samples = data2d.shape
    assert n_items == irt_probs.shape[0]
    k_range = np.arange(1, n_samples + 1)
    per_item = []
    for i in range(n_items):
        per_item.append([compute_pass_iatk_irt(irt_probs[i], k) for k in k_range])
    return np.nanmean(np.vstack(per_item), axis=0)

# ---------------------------
# Generate distributions
# ---------------------------
rng = np.random.default_rng(0)
n_items, K_max = 5000, 100
dummy_X = np.zeros((n_items, K_max))

dists = {
    "Left-tail (Beta(1,5))": rng.beta(1, 5, size=n_items),
    "Right-tail (Beta(5,1))": rng.beta(5, 1, size=n_items),
    "Both-tails (Beta(0.5,0.5))": rng.beta(0.5, 0.5, size=n_items),
    "Uniform (Beta(1,1))": rng.beta(1, 1, size=n_items),
    "Normal-like (Beta(20,20))": rng.beta(20, 20, size=n_items),
}

# ---------------------------
# Compute curves
# ---------------------------
k_range = np.arange(1, K_max + 1)
curves = {}
for name, probs in dists.items():
    curves[name] = compute_pass_datk_irt(dummy_X, probs)

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(2, 1, figsize=(7, 10), sharex=False)

# Top: pass@k vs k (linear scale)
for name, y in curves.items():
    axes[0].plot(k_range, y, linewidth=2, label=name)
axes[0].set_xlabel("k", fontsize=12)
axes[0].set_ylabel("pass@k", fontsize=12)
axes[0].set_title("pass@k vs k (linear scale)", fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9)

# Bottom: -log(pass@k) vs k (log-log scale)
for name, y in curves.items():
    y_clipped = np.clip(y, 1e-12, 1.0)  # avoid log(0)
    axes[1].plot(k_range, -np.log(y_clipped), linewidth=2, label=name)
axes[1].set_xlabel("k", fontsize=12)
axes[1].set_ylabel("-log(pass@k)", fontsize=12)
axes[1].set_title("-log(pass@k) vs k (log-log scale)", fontsize=14)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("passatk_and_loglog.png", dpi=300, bbox_inches="tight")