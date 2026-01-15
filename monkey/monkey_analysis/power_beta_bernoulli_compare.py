import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, betaln
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(7)

def rasch_prob(theta, b):
    return expit(theta[:, None] - b[None, :])

def simulate_binary(theta, b):
    p = rasch_prob(theta, b)
    y = rng.binomial(1, p)
    return y, p

def simulate_prob_matrix(theta, b, noise_sd=0.01):
    p = rasch_prob(theta, b)
    noisy = p + rng.normal(0.0, noise_sd, size=p.shape)
    noisy = np.clip(noisy, 1e-6, 1 - 1e-6)
    return noisy, p

def fit_rasch_bernoulli(Y, maxiter=4000):
    M, N = Y.shape
    x0 = np.zeros(M + N)

    def unpack(x):
        theta = x[:M]
        b = x[M:]
        # constrain mean(b)=0 by recentering; preserves theta-b
        shift = b.mean()
        b = b - shift
        theta = theta - shift
        return theta, b

    def nll(x):
        theta, b = unpack(x)
        p = rasch_prob(theta, b)
        eps = 1e-12
        ll = np.sum(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))
        return -ll

    def grad(x):
        theta, b = unpack(x)
        p = rasch_prob(theta, b)
        g_theta = np.sum(p - Y, axis=1)
        g_b = -np.sum(p - Y, axis=0)
        return np.concatenate([g_theta, g_b])

    res = minimize(nll, x0, jac=grad, method="L-BFGS-B", options={"maxiter": maxiter})
    return unpack(res.x), res

def fit_rasch_beta(Pobs, phi=400.0, maxiter=6000):
    M, N = Pobs.shape
    x0 = np.zeros(M + N)
    eps = 1e-6
    Pobs = np.clip(Pobs, eps, 1 - eps)

    def unpack(x):
        theta = x[:M]
        b = x[M:]
        shift = b.mean()
        b = b - shift
        theta = theta - shift
        return theta, b

    def nll(x):
        theta, b = unpack(x)
        p = rasch_prob(theta, b)
        p = np.clip(p, eps, 1 - eps)
        a = phi * p
        bb = phi * (1 - p)
        ll = np.sum((a - 1) * np.log(Pobs) + (bb - 1) * np.log(1 - Pobs) - betaln(a, bb))
        return -ll

    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": maxiter})
    return unpack(res.x), res

def recovery(true_b, est_b):
    true_b = true_b - true_b.mean()
    est_b = est_b - est_b.mean()
    rmse = float(np.sqrt(np.mean((est_b - true_b) ** 2)))
    corr = float(np.corrcoef(true_b, est_b)[0, 1])
    return rmse, corr

# --- simulate over range of M values with multiple repetitions ---
N = 100
n_reps = 10  # Number of repetitions for averaging

# Range of test taker counts to evaluate: 2^n for n=1 to 7
M_values = [2, 4, 8, 16, 32, 64, 128]

# Store results: shape (n_reps, len(M_values))
rmse_binary_all = np.zeros((n_reps, len(M_values)))
rmse_beta_all = np.zeros((n_reps, len(M_values)))
corr_binary_all = np.zeros((n_reps, len(M_values)))
corr_beta_all = np.zeros((n_reps, len(M_values)))

print(f"Running {n_reps} repetitions for each M value...")
for rep in range(n_reps):
    # Generate new true parameters for each repetition
    b_true = rng.normal(0, 1, size=N)

    for i, M in enumerate(M_values):
        theta = rng.normal(0, 1, size=M)

        # Binary Rasch
        Y, _ = simulate_binary(theta, b_true)
        (_, b_hat_bin), _ = fit_rasch_bernoulli(Y)
        rmse_b, corr_b = recovery(b_true, b_hat_bin)
        rmse_binary_all[rep, i] = rmse_b
        corr_binary_all[rep, i] = corr_b

        # Beta Rasch
        P_obs, _ = simulate_prob_matrix(theta, b_true, noise_sd=0.01)
        (_, b_hat_beta), _ = fit_rasch_beta(P_obs, phi=400.0)
        rmse_be, corr_be = recovery(b_true, b_hat_beta)
        rmse_beta_all[rep, i] = rmse_be
        corr_beta_all[rep, i] = corr_be

    if (rep + 1) % 10 == 0:
        print(f"  Completed {rep + 1}/{n_reps} repetitions")

# Compute mean and std
rmse_binary_mean = rmse_binary_all.mean(axis=0)
rmse_binary_std = rmse_binary_all.std(axis=0)
rmse_beta_mean = rmse_beta_all.mean(axis=0)
rmse_beta_std = rmse_beta_all.std(axis=0)

corr_binary_mean = corr_binary_all.mean(axis=0)
corr_binary_std = corr_binary_all.std(axis=0)
corr_beta_mean = corr_beta_all.mean(axis=0)
corr_beta_std = corr_beta_all.std(axis=0)

# Print summary
print(f"\nResults (mean ± std over {n_reps} repetitions):")
print("-" * 80)
for i, M in enumerate(M_values):
    print(f"M={M:4d}:  Binary RMSE={rmse_binary_mean[i]:.4f}±{rmse_binary_std[i]:.4f}  |  "
          f"Beta RMSE={rmse_beta_mean[i]:.4f}±{rmse_beta_std[i]:.4f}")

# Save results to file for quick reloading
output_dir = "../../result/monkey_analysis"
os.makedirs(output_dir, exist_ok=True)
data_path = f"{output_dir}/power_beta_bernoulli_data.npz"
np.savez(data_path,
         M_values=np.array(M_values),
         rmse_binary_all=rmse_binary_all,
         rmse_beta_all=rmse_beta_all,
         corr_binary_all=corr_binary_all,
         corr_beta_all=corr_beta_all,
         rmse_binary_mean=rmse_binary_mean,
         rmse_binary_std=rmse_binary_std,
         rmse_beta_mean=rmse_beta_mean,
         rmse_beta_std=rmse_beta_std,
         corr_binary_mean=corr_binary_mean,
         corr_binary_std=corr_binary_std,
         corr_beta_mean=corr_beta_mean,
         corr_beta_std=corr_beta_std,
         n_reps=n_reps,
         N=N)
print(f"\nData saved to: {data_path}")

# ICML-ready matplotlib configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 14
})

# Create figure with two panels side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

M_arr = np.array(M_values)

# Colors: blue for Binary, pink/magenta for Beta
color_binary = '#4169E1'  # Royal blue
color_beta = '#E91E63'    # Pink/magenta

# Panel 1: RMSE vs M with error bars (linear X-axis)
ax = axes[0]
ax.errorbar(M_arr, rmse_binary_mean, yerr=rmse_binary_std, fmt='o-', color=color_binary,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Binary Rasch')
ax.errorbar(M_arr, rmse_beta_mean, yerr=rmse_beta_std, fmt='s-', color=color_beta,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Beta Rasch')
ax.set_xlabel('Number of Test Takers (M)')
ax.set_ylabel('RMSE of Item Difficulty (b)')
ax.set_title('Parameter Recovery vs. Sample Size (1PL)')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Panel 2: Correlation vs M with error bars (linear X-axis)
ax = axes[1]
ax.errorbar(M_arr, corr_binary_mean, yerr=corr_binary_std, fmt='o-', color=color_binary,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Binary Rasch')
ax.errorbar(M_arr, corr_beta_mean, yerr=corr_beta_std, fmt='s-', color=color_beta,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Beta Rasch')
ax.set_xlabel('Number of Test Takers (M)')
ax.set_ylabel('Correlation (r)')
ax.set_title('Correlation vs. Sample Size (1PL)')
ax.legend(loc='lower right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])
ax.set_ylim(0, 1.05)

plt.tight_layout()

# Save figure
output_path = f"{output_dir}/power_beta_bernoulli_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")
