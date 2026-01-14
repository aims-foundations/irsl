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

# --- simulate ---
M_full, N = 1000, 100
theta_full = rng.normal(0, 1, size=M_full)
b_true = rng.normal(0, 1, size=N)

Y, _ = simulate_binary(theta_full, b_true)
(theta_hat_bin, b_hat_bin), _ = fit_rasch_bernoulli(Y)
rmse_bin, corr_bin = recovery(b_true, b_hat_bin)

M_small = 4
theta_small = rng.normal(0, 1, size=M_small)
P_obs, _ = simulate_prob_matrix(theta_small, b_true, noise_sd=0.01)
(theta_hat_beta, b_hat_beta), _ = fit_rasch_beta(P_obs, phi=400.0)
rmse_beta, corr_beta = recovery(b_true, b_hat_beta)

# Also fit Binary Rasch with M=4 for comparison
Y_small, _ = simulate_binary(theta_small, b_true)
(theta_hat_bin_small, b_hat_bin_small), _ = fit_rasch_bernoulli(Y_small)
rmse_bin_small, corr_bin_small = recovery(b_true, b_hat_bin_small)

print("Binary (M=1000):  RMSE=", rmse_bin, " Corr=", corr_bin)
print("Binary (M=4):     RMSE=", rmse_bin_small, " Corr=", corr_bin_small)
print("Prob   (M=4):     RMSE=", rmse_beta, " Corr=", corr_beta)

# ICML-ready matplotlib configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': True,
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})

# Prepare centered parameters for plotting
b_true_centered = b_true - b_true.mean()
b_hat_bin_centered = b_hat_bin - b_hat_bin.mean()
b_hat_bin_small_centered = b_hat_bin_small - b_hat_bin_small.mean()
b_hat_beta_centered = b_hat_beta - b_hat_beta.mean()

# Create 2x3 panel figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Panel 1 (top-left): Binary method scatter plot
ax = axes[0, 0]
ax.scatter(b_true_centered, b_hat_bin_centered, alpha=0.6, s=30, color='steelblue')
lim_min = min(b_true_centered.min(), b_hat_bin_centered.min()) - 0.2
lim_max = max(b_true_centered.max(), b_hat_bin_centered.max()) + 0.2
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1, label='Identity')
ax.set_xlabel(r'True $b$ (centered)')
ax.set_ylabel(r'Estimated $b$ (centered)')
ax.set_title(r'Binary Rasch ($M=1000$)')
ax.text(0.05, 0.95, f'RMSE={rmse_bin:.4f}\n$r$={corr_bin:.4f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_aspect('equal')

# Panel 2 (top-middle): Binary method with M=4 scatter plot
ax = axes[0, 1]
ax.scatter(b_true_centered, b_hat_bin_small_centered, alpha=0.6, s=30, color='darkgreen')
lim_min = min(b_true_centered.min(), b_hat_bin_small_centered.min()) - 0.2
lim_max = max(b_true_centered.max(), b_hat_bin_small_centered.max()) + 0.2
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1, label='Identity')
ax.set_xlabel(r'True $b$ (centered)')
ax.set_ylabel(r'Estimated $b$ (centered)')
ax.set_title(r'Binary Rasch ($M=4$)')
ax.text(0.05, 0.95, f'RMSE={rmse_bin_small:.4f}\n$r$={corr_bin_small:.4f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_aspect('equal')

# Panel 3 (top-right): Beta method scatter plot
ax = axes[0, 2]
ax.scatter(b_true_centered, b_hat_beta_centered, alpha=0.6, s=30, color='coral')
lim_min = min(b_true_centered.min(), b_hat_beta_centered.min()) - 0.2
lim_max = max(b_true_centered.max(), b_hat_beta_centered.max()) + 0.2
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1, label='Identity')
ax.set_xlabel(r'True $b$ (centered)')
ax.set_ylabel(r'Estimated $b$ (centered)')
ax.set_title(r'Beta Rasch ($M=4$, $\phi=400$)')
ax.text(0.05, 0.95, f'RMSE={rmse_beta:.4f}\n$r$={corr_beta:.4f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_aspect('equal')

# Panel 4 (bottom-left): Binary M=1000 residuals
ax = axes[1, 0]
residuals_bin = b_hat_bin_centered - b_true_centered
ax.scatter(b_true_centered, residuals_bin, alpha=0.6, s=30, color='steelblue')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel(r'True $b$ (centered)')
ax.set_ylabel(r'Residual (Est - True)')
ax.set_title(r'Binary Rasch ($M=1000$) Residuals')
ax.grid(True, alpha=0.3)

# Panel 5 (bottom-middle): Binary M=4 residuals
ax = axes[1, 1]
residuals_bin_small = b_hat_bin_small_centered - b_true_centered
ax.scatter(b_true_centered, residuals_bin_small, alpha=0.6, s=30, color='darkgreen')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel(r'True $b$ (centered)')
ax.set_ylabel(r'Residual (Est - True)')
ax.set_title(r'Binary Rasch ($M=4$) Residuals')
ax.grid(True, alpha=0.3)

# Panel 6 (bottom-right): Beta method residuals
ax = axes[1, 2]
residuals_beta = b_hat_beta_centered - b_true_centered
ax.scatter(b_true_centered, residuals_beta, alpha=0.6, s=30, color='coral')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel(r'True $b$ (centered)')
ax.set_ylabel(r'Residual (Est - True)')
ax.set_title(r'Beta Rasch ($M=4$) Residuals')
ax.grid(True, alpha=0.3)

# Align y-axis limits for residual plots
all_residuals = np.concatenate([residuals_bin, residuals_bin_small, residuals_beta])
res_lim = max(abs(all_residuals.min()), abs(all_residuals.max())) * 1.1
axes[1, 0].set_ylim(-res_lim, res_lim)
axes[1, 1].set_ylim(-res_lim, res_lim)
axes[1, 2].set_ylim(-res_lim, res_lim)

plt.tight_layout()

# Save figure
output_dir = "../../result/monkey_analysis"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/power_beta_bernoulli_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")
