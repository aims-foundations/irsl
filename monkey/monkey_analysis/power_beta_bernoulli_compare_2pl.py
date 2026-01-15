import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, betaln
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)

def prob_2pl(theta, a, b):
    """2PL probability: p = sigmoid(a * (theta - b))"""
    return expit(a[None, :] * (theta[:, None] - b[None, :]))

def simulate_binary_2pl(theta, a, b):
    """Simulate binary responses from 2PL model"""
    p = prob_2pl(theta, a, b)
    y = rng.binomial(1, p)
    return y, p

def simulate_prob_matrix_2pl(theta, a, b, noise_sd=0.01):
    """Simulate noisy probability matrix from 2PL model"""
    p = prob_2pl(theta, a, b)
    noisy = p + rng.normal(0.0, noise_sd, size=p.shape)
    noisy = np.clip(noisy, 1e-6, 1 - 1e-6)
    return noisy, p

def fit_2pl_bernoulli(Y, maxiter=6000):
    """Fit 2PL model to binary responses using MLE

    Improved with bounded discrimination and better initialization.
    """
    M, N = Y.shape
    eps = 1e-12

    # Better initialization from observed response rates
    p_obs = Y.mean(axis=0)  # Item pass rates
    p_obs = np.clip(p_obs, 0.01, 0.99)
    b_init = -np.log(p_obs / (1 - p_obs))  # Approximate difficulty
    b_init = b_init - b_init.mean()

    theta_obs = Y.mean(axis=1)  # Person success rates
    theta_obs = np.clip(theta_obs, 0.01, 0.99)
    theta_init = np.log(theta_obs / (1 - theta_obs))
    theta_init = theta_init - theta_init.mean()

    def unpack(x):
        theta = x[:M]
        b = x[M:M + N]
        log_a = x[M + N:]
        log_a = np.clip(log_a, -1.5, 1.5)
        a = np.exp(log_a)
        shift = b.mean()
        b = b - shift
        theta = theta - shift
        return theta, a, b

    def nll(x):
        theta, a, b = unpack(x)
        p = prob_2pl(theta, a, b)
        ll = np.sum(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))
        # Light regularization on log_a
        log_a = x[M + N:]
        reg = 0.1 * np.sum(log_a ** 2)
        return -ll + reg

    bounds = [(None, None)] * (M + N) + [(-1.5, 1.5)] * N

    # Data-driven initialization (single run)
    x0 = np.concatenate([theta_init, b_init, np.zeros(N)])

    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                  options={"maxiter": maxiter})

    return unpack(res.x), res

def fit_2pl_beta(Pobs, phi=400.0, maxiter=8000):
    """Fit 2PL model to probability matrix using Beta likelihood

    Improved optimization with:
    1. Better initialization using observed data
    2. Bounded discrimination parameters
    3. L2 regularization on discrimination
    """
    M, N = Pobs.shape
    eps = 1e-6
    Pobs = np.clip(Pobs, eps, 1 - eps)

    # Better initialization: estimate theta and b from observed probabilities
    # Using the fact that logit(p) ≈ a*(theta - b)
    logit_P = np.log(Pobs / (1 - Pobs))
    theta_init = logit_P.mean(axis=1)  # Average logit across items
    b_init = -logit_P.mean(axis=0)  # Average logit across people (negated)
    theta_init = theta_init - theta_init.mean()
    b_init = b_init - b_init.mean()

    def unpack(x):
        theta = x[:M]
        b = x[M:M + N]
        log_a = x[M + N:]
        # Clip log_a to prevent extreme values
        log_a = np.clip(log_a, -1.5, 1.5)  # a in [0.22, 4.48]
        a = np.exp(log_a)
        shift = b.mean()
        b = b - shift
        theta = theta - shift
        return theta, a, b

    def nll(x):
        theta, a, b = unpack(x)
        p = prob_2pl(theta, a, b)
        p = np.clip(p, eps, 1 - eps)
        alpha = phi * p
        beta_param = phi * (1 - p)
        ll = np.sum((alpha - 1) * np.log(Pobs) + (beta_param - 1) * np.log(1 - Pobs) - betaln(alpha, beta_param))
        # L2 regularization on log_a to keep discrimination near 1
        log_a = x[M + N:]
        reg = 0.5 * np.sum(log_a ** 2)  # Stronger regularization
        return -ll + reg

    # Bounds: constrain log_a to reasonable range
    bounds = [(None, None)] * (M + N) + [(-1.5, 1.5)] * N

    # Data-driven initialization (single run)
    x0 = np.concatenate([theta_init, b_init, np.zeros(N)])

    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                  options={"maxiter": maxiter, "ftol": 1e-10})

    return unpack(res.x), res

def recovery(true_vals, est_vals):
    """Compute RMSE and correlation for parameter recovery"""
    true_centered = true_vals - true_vals.mean()
    est_centered = est_vals - est_vals.mean()
    rmse = float(np.sqrt(np.mean((est_centered - true_centered) ** 2)))
    corr = float(np.corrcoef(true_centered, est_centered)[0, 1])
    return rmse, corr

def recovery_a(true_a, est_a):
    """Compute RMSE and correlation for discrimination parameter recovery (no centering)"""
    rmse = float(np.sqrt(np.mean((est_a - true_a) ** 2)))
    corr = float(np.corrcoef(true_a, est_a)[0, 1])
    return rmse, corr

# --- Simulation over range of M values with multiple repetitions ---
N = 100
n_reps = 10  # Reduced for faster runtime

# Range of test taker counts to evaluate: 2^n for n=1 to 7
M_values = [2, 4, 8, 16, 32, 64, 128]

# Store results for b parameter
rmse_binary_b_all = np.zeros((n_reps, len(M_values)))
rmse_beta_b_all = np.zeros((n_reps, len(M_values)))
corr_binary_b_all = np.zeros((n_reps, len(M_values)))
corr_beta_b_all = np.zeros((n_reps, len(M_values)))

# Store results for a parameter
rmse_binary_a_all = np.zeros((n_reps, len(M_values)))
rmse_beta_a_all = np.zeros((n_reps, len(M_values)))
corr_binary_a_all = np.zeros((n_reps, len(M_values)))
corr_beta_a_all = np.zeros((n_reps, len(M_values)))

print(f"Running {n_reps} repetitions for each M value (2PL model)...")
for rep in range(n_reps):
    # Generate true parameters for each repetition
    b_true = rng.normal(0, 1, size=N)
    # Generate discrimination parameters from log-normal (ensures a > 0, centered around 1)
    a_true = rng.lognormal(0, 0.5, size=N)

    for i, M in enumerate(M_values):
        theta = rng.normal(0, 1, size=M)

        # Binary 2PL
        Y, _ = simulate_binary_2pl(theta, a_true, b_true)
        (_, a_hat_bin, b_hat_bin), _ = fit_2pl_bernoulli(Y)
        rmse_b, corr_b = recovery(b_true, b_hat_bin)
        rmse_a, corr_a = recovery_a(a_true, a_hat_bin)
        rmse_binary_b_all[rep, i] = rmse_b
        corr_binary_b_all[rep, i] = corr_b
        rmse_binary_a_all[rep, i] = rmse_a
        corr_binary_a_all[rep, i] = corr_a

        # Beta 2PL
        P_obs, _ = simulate_prob_matrix_2pl(theta, a_true, b_true, noise_sd=0.01)
        (_, a_hat_beta, b_hat_beta), _ = fit_2pl_beta(P_obs, phi=400.0)
        rmse_b, corr_b = recovery(b_true, b_hat_beta)
        rmse_a, corr_a = recovery_a(a_true, a_hat_beta)
        rmse_beta_b_all[rep, i] = rmse_b
        corr_beta_b_all[rep, i] = corr_b
        rmse_beta_a_all[rep, i] = rmse_a
        corr_beta_a_all[rep, i] = corr_a

    if (rep + 1) % 5 == 0:
        print(f"  Completed {rep + 1}/{n_reps} repetitions")

# Compute mean and std for b
rmse_binary_b_mean = rmse_binary_b_all.mean(axis=0)
rmse_binary_b_std = rmse_binary_b_all.std(axis=0)
rmse_beta_b_mean = rmse_beta_b_all.mean(axis=0)
rmse_beta_b_std = rmse_beta_b_all.std(axis=0)
corr_binary_b_mean = corr_binary_b_all.mean(axis=0)
corr_binary_b_std = corr_binary_b_all.std(axis=0)
corr_beta_b_mean = corr_beta_b_all.mean(axis=0)
corr_beta_b_std = corr_beta_b_all.std(axis=0)

# Compute mean and std for a
rmse_binary_a_mean = rmse_binary_a_all.mean(axis=0)
rmse_binary_a_std = rmse_binary_a_all.std(axis=0)
rmse_beta_a_mean = rmse_beta_a_all.mean(axis=0)
rmse_beta_a_std = rmse_beta_a_all.std(axis=0)
corr_binary_a_mean = corr_binary_a_all.mean(axis=0)
corr_binary_a_std = corr_binary_a_all.std(axis=0)
corr_beta_a_mean = corr_beta_a_all.mean(axis=0)
corr_beta_a_std = corr_beta_a_all.std(axis=0)

# Print summary
print(f"\nResults (mean +/- std over {n_reps} repetitions):")
print("-" * 100)
print("Difficulty (b) recovery:")
for i, M in enumerate(M_values):
    print(f"M={M:4d}:  Binary RMSE={rmse_binary_b_mean[i]:.4f}+/-{rmse_binary_b_std[i]:.4f}  |  "
          f"Beta RMSE={rmse_beta_b_mean[i]:.4f}+/-{rmse_beta_b_std[i]:.4f}")
print("\nDiscrimination (a) recovery:")
for i, M in enumerate(M_values):
    print(f"M={M:4d}:  Binary RMSE={rmse_binary_a_mean[i]:.4f}+/-{rmse_binary_a_std[i]:.4f}  |  "
          f"Beta RMSE={rmse_beta_a_mean[i]:.4f}+/-{rmse_beta_a_std[i]:.4f}")

# Save results
output_dir = "../../result/monkey_analysis"
os.makedirs(output_dir, exist_ok=True)
data_path = f"{output_dir}/power_beta_bernoulli_2pl_data.npz"
np.savez(data_path,
         M_values=np.array(M_values),
         rmse_binary_b_all=rmse_binary_b_all,
         rmse_beta_b_all=rmse_beta_b_all,
         corr_binary_b_all=corr_binary_b_all,
         corr_beta_b_all=corr_beta_b_all,
         rmse_binary_a_all=rmse_binary_a_all,
         rmse_beta_a_all=rmse_beta_a_all,
         corr_binary_a_all=corr_binary_a_all,
         corr_beta_a_all=corr_beta_a_all,
         rmse_binary_b_mean=rmse_binary_b_mean,
         rmse_binary_b_std=rmse_binary_b_std,
         rmse_beta_b_mean=rmse_beta_b_mean,
         rmse_beta_b_std=rmse_beta_b_std,
         rmse_binary_a_mean=rmse_binary_a_mean,
         rmse_binary_a_std=rmse_binary_a_std,
         rmse_beta_a_mean=rmse_beta_a_mean,
         rmse_beta_a_std=rmse_beta_a_std,
         corr_binary_b_mean=corr_binary_b_mean,
         corr_binary_b_std=corr_binary_b_std,
         corr_beta_b_mean=corr_beta_b_mean,
         corr_beta_b_std=corr_beta_b_std,
         corr_binary_a_mean=corr_binary_a_mean,
         corr_binary_a_std=corr_binary_a_std,
         corr_beta_a_mean=corr_beta_a_mean,
         corr_beta_a_std=corr_beta_a_std,
         n_reps=n_reps,
         N=N)
print(f"\nData saved to: {data_path}")

# Plotting
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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
M_arr = np.array(M_values)

# Colors: blue for Binary, pink/magenta for Beta, purple available if needed
color_binary = '#4169E1'  # Royal blue
color_beta = '#E91E63'    # Pink/magenta

# Panel 1: RMSE of b vs M
ax = axes[0]
ax.errorbar(M_arr, rmse_binary_b_mean, yerr=rmse_binary_b_std, fmt='o-', color=color_binary,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Binary 2PL')
ax.errorbar(M_arr, rmse_beta_b_mean, yerr=rmse_beta_b_std, fmt='s-', color=color_beta,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Beta 2PL')
ax.set_xlabel('Number of Test Takers (M)')
ax.set_ylabel('RMSE of Item Difficulty (b)')
ax.set_title('Difficulty Recovery vs. Sample Size (2PL)')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Panel 2: Correlation of b vs M
ax = axes[1]
ax.errorbar(M_arr, corr_binary_b_mean, yerr=corr_binary_b_std, fmt='o-', color=color_binary,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Binary 2PL')
ax.errorbar(M_arr, corr_beta_b_mean, yerr=corr_beta_b_std, fmt='s-', color=color_beta,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Beta 2PL')
ax.set_xlabel('Number of Test Takers (M)')
ax.set_ylabel('Correlation (r)')
ax.set_title('Correlation vs. Sample Size (2PL)')
ax.legend(loc='lower right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])
ax.set_ylim(0, 1.05)

plt.tight_layout()
output_path = f"{output_dir}/power_beta_bernoulli_2pl_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")
