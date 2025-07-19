import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

# 1. Download and load data
data_path = snapshot_download(repo_id="stair-lab/monkey_3d_data", repo_type="dataset")
loaded = torch.load(os.path.join(data_path, "gsm_tensor.pth"), map_location="cpu")
scores = loaded["data_tensor"]    # shape [M, Q, S]
models = loaded["models"]         # length M

M, Q, S = scores.shape

# 2. Set up a single‑column grid
nrows = M
fig, axes = plt.subplots(nrows, 1, figsize=(6, nrows * 3), constrained_layout=True)
axes = axes.flatten()

# 3. Plot one histogram per model using nanmean
for i in range(M):
    # compute per‑question average ignoring NaNs
    # convert to NumPy for np.nanmean
    arr = scores[i].numpy()     # shape (Q, S)
    avg_per_question = np.nanmean(arr, axis=1)  # shape (Q,)
    
    ax = axes[i]
    ax.hist(avg_per_question, bins=20, edgecolor="black")
    ax.set_title(models[i], fontsize=10)
    ax.set_xlabel("Avg probability (over samples, NaNs ignored)")
    ax.set_ylabel("Number of questions")
    ax.set_xlim(0, 1)

# 4. Save figure to file
plt.savefig("model_histograms.png", dpi=300, bbox_inches="tight")
