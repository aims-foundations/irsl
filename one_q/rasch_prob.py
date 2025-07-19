import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.optim import LBFGS
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import snapshot_download


# Compute MSE on train & test
def compute_mse(preds, data, train_mask, test_mask):
    # flatten masked entries
    tr_p = preds[train_mask.bool()];  tr_y = data[train_mask.bool()]
    te_p = preds[test_mask.bool()];   te_y = data[test_mask.bool()]
    train_mse = F.mse_loss(tr_p, tr_y)
    test_mse  = F.mse_loss(te_p, te_y)
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test  MSE: {test_mse:.6f}")
    return train_mse, test_mse

# trainer function unchanged
def trainer(parameters, optim, closure, n_iter=1000, verbose=True):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for it in pbar:
        if it > 0:
            prev = [p.clone() for p in parameters]
            prev_loss = loss.clone()
        loss = optim.step(closure)
        if it > 0:
            dloss = (prev_loss - loss).item()
            dparam = sum(torch.norm(a-b).item() for a,b in zip(prev,parameters))
            gradn = sum(torch.norm(p.grad).item() for p in parameters if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": gradn, "d_param": dparam, "d_loss": dloss})
            if dloss < 1e-5 and dparam < 1e-5 and gradn < 1e-5:
                break
    return parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1–5. (Same as before: load, filter, NaN cleanup)
platinum = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
filtered_platinum = platinum.filter(lambda item: item["cleaning_status"] in ["consensus", "verified"])
platinum_qs = set(filtered_platinum["question"])

data_path = snapshot_download(repo_id="stair-lab/monkey_3d_data", repo_type="dataset")
scores = torch.load(os.path.join(data_path, "gsm_tensor.pth"), map_location="cpu")["data_tensor"]  # [M,Q,S]
models = torch.load(os.path.join(data_path, "gsm_tensor.pth"), map_location="cpu")["models"]
questions = torch.load(os.path.join(data_path, "gsm_tensor.pth"), map_location="cpu")["questions"]

# Exclude and mask models/questions, drop all-NaN slices (as you had)
exclude = {"Meta-Llama-3-8B-Instruct","Pythia_6.9B","Meta-Llama-3-70B-Instruct",
           "gemma-3-27b-it","Mistral-7B-v0.1","Pythia_12B"}
mask_m = [m not in exclude for m in models]
scores = scores[mask_m]
models = [m for m in models if m not in exclude]

mask_q = [q in platinum_qs for q in questions]
scores = scores[:, mask_q]
questions = [q for q,k in zip(questions,mask_q) if k]

keep_q = ~torch.all(torch.isnan(scores), dim=(0,2))
scores = scores[:, keep_q]
questions = [q for q,k in zip(questions,keep_q) if k]

keep_s = ~torch.all(torch.isnan(scores), dim=(0,1))
scores = scores[:, :, keep_s]

print("After cleanup:", scores.shape)  # should be [M, Q, S]

# --- NEW: compute your continuous response matrix [M, Q] ---
# average over samples (last dim), ignoring NaNs
resp_matrix = np.nanmean(scores.numpy(), axis=2)  # shape: (M, Q)

# Prepare data for modeling
data_withnan = torch.tensor(resp_matrix, dtype=torch.float, device=device)
data_withneg1 = data_withnan.nan_to_num(nan=-1.0)
data_idtor = (data_withneg1 != -1).to(torch.float)  # mask of observed entries
data = data_withneg1 * data_idtor                  # unobserved now zeroed

n_takers, n_items = data.shape

# Split train/test masks
valid = False
while not valid:
    train_idtor = torch.bernoulli(data_idtor * 0.8).int()
    test_idtor = (data_idtor - train_idtor).int()
    valid = (train_idtor.sum(dim=1) != 0).all() and (train_idtor.sum(dim=0) != 0).all()

# Initialize nuisance thetas and item parameters zs
B = 50000
optimized_zs = []
thetas_nuisance = torch.randn(150, n_takers, device=device)

for i in tqdm(range(0, n_items, B)):
    batch = data[:, i:i+B]
    mask_batch = train_idtor[:, i:i+B]
    curr_B = batch.shape[1]

    z = torch.randn(curr_B, requires_grad=True, device=device)
    optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    def closure_z():
        optim_z.zero_grad()
        preds = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
        loss = ((preds - batch)**2 * mask_batch).sum() / mask_batch.sum()
        loss.backward()
        return loss

    z_opt = trainer([z], optim_z, closure_z)[0].detach()
    optimized_zs.append(z_opt)

zs = torch.cat(optimized_zs)  # final item effects

# Fit thetas
thetas = torch.randn(n_takers, requires_grad=True, device=device)
optim_th = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

def closure_th():
    optim_th.zero_grad()
    preds = torch.sigmoid(thetas[:, None] + zs[None, :])
    loss = ((preds - data)**2 * train_idtor).sum() / train_idtor.sum()
    loss.backward()
    return loss

thetas = trainer([thetas], optim_th, closure_th)[0]

all_preds = torch.sigmoid(thetas[:, None] + zs[None, :])
train_mse, test_mse = compute_mse(all_preds, data, train_idtor, test_idtor)

rand_preds = torch.rand_like(all_preds, device=device)
rand_train_mse, rand_test_mse = compute_mse(rand_preds, data, train_idtor, test_idtor)

import matplotlib.pyplot as plt

# Convert to numpy
orig = resp_matrix
recon = all_preds.detach().cpu().numpy()

# 1×2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plot original
im0 = axes[0].imshow(orig, aspect='auto', cmap='RdBu', vmin=0, vmax=1)
axes[0].set_title('Original Response')
axes[0].set_xlabel('Question index')
axes[0].set_yticks(np.arange(len(models)))
axes[0].set_yticklabels(models)
axes[0].set_ylabel('Model')

# Plot reconstruction
im1 = axes[1].imshow(recon, aspect='auto', cmap='RdBu', vmin=0, vmax=1)
axes[1].set_title('Reconstructed Response')
axes[1].set_yticks(np.arange(len(models)))
axes[1].set_yticklabels(models)

# Make room on the right: 0.85 means subplots occupy [0:0.85] of figure width
fig.subplots_adjust(right=0.85)

# Create new axes at [left, bottom, width, height] in figure coordinates
cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # tweak these numbers as needed

# Draw a single colorbar in that axes
fig.colorbar(im1, cax=cax, label='Response value')

# Save and show
fig.savefig('response_matrices.png', dpi=300, bbox_inches='tight')

# Convert predictions to numpy
orig = resp_matrix                     # shape (M, Q)
pred = all_preds.detach().cpu().numpy() # shape (M, Q)

M, Q = orig.shape
x = np.arange(1, Q+1)  # questions 1…Q

# Pick the first 5 takers (or any 5 indices you like)
indices = list(range(M))

fig, axes = plt.subplots(len(indices), 1, figsize=(10, 2*len(indices)), sharex=True, sharey=True)

for ax, i in zip(axes, indices):
    ax.plot(x, orig[i],   label='Ground truth', linewidth=1)
    ax.plot(x, pred[i],   label='Prediction',   linewidth=1, linestyle='--')
    ax.set_ylim(0, 1)
    ax.set_ylabel(f'{models[i]} Probs')
    ax.legend(loc='upper right', fontsize='small')

axes[-1].set_xlabel('Question index')
plt.tight_layout()

# save to file
fig.savefig('taker_curves.png', dpi=300, bbox_inches='tight')

# --- ERROR DISTRIBUTION PLOTS ---

# Compute error matrix (numpy)
errors = orig - pred   # shape (M, Q)

indices = list(range(M))     # for all M takers

# Create one row per taker
fig, axes = plt.subplots(len(indices), 1, figsize=(10, 2*len(indices)), sharex=True)

for ax, i in zip(axes, indices):
    ax.hist(errors[i], bins=30, range=(-1, 1), alpha=0.7)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_ylabel(f'{models[i]} Error')
    ax.set_yticks([])  # optional: clean up y‑axis

# Common x‑label
axes[-1].set_xlabel('Error (GT – Pred)')

plt.tight_layout()

# Save and show
fig.savefig('error_distributions.png', dpi=300, bbox_inches='tight')