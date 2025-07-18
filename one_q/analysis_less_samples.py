import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt

# 1. Load GSM8K platinum questions
platinum = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
valid = platinum.filter(lambda ex: ex["cleaning_status"] in ["consensus", "verified"])
platinum_questions = set(valid["question"])

# 2. Download and locate JSON files in your HF dataset
repo_dir = Path(snapshot_download(
    repo_id="stair-lab/one_question_less_samples",
    repo_type="dataset"
))
json_files = sorted(repo_dir.glob("*.json"))

# 3. Load each model’s records into a DataFrame
model_dfs = {}
for jf in json_files:
    with open(jf, "r", encoding="utf-8") as f:
        records = json.load(f)
    # filenames like "ModelName_gsm.json"
    model_name = jf.stem.split("_")[0]
    df = pd.DataFrame(records)  # columns: ["question", "is_corrects"]
    model_dfs[model_name] = df

# 4. Find intersection of questions across all models
all_sets = [set(df["question"]) for df in model_dfs.values()]
common_qs = set.intersection(*all_sets)

# 5. Further restrict to platinum questions
common_plat = common_qs & platinum_questions

# 6. Filter each DataFrame down to that set
for name, df in model_dfs.items():
    df = df[df["question"].isin(common_plat)].copy()
    df.reset_index(drop=True, inplace=True)
    model_dfs[name] = df

# 7. Compute per-question means & ranks, then overall mean rank, and plot
#   a) For each model & question, compute mean over 50 samples
#   b) For each question, rank models by that mean (1 = best)
#   c) Compute each model’s overall mean-of-means and rank them
#   d) Combine per-question ranks + overall rank into one matrix
#   e) Visualize with imshow + text annotations

# a) build a [models × questions] array of mean accuracies
model_names = list(model_dfs.keys())
num_models  = len(model_names)
num_q       = len(common_plat)

means_mat = np.zeros((num_models, num_q), dtype=float)
for i, name in enumerate(model_names):
    df = model_dfs[name]
    # mean over the 50 samples
    means_mat[i, :] = np.array([np.nanmean(lst) for lst in df["is_corrects"]])

# b) per-question ranking (1 = highest mean)
question_ranks = np.zeros_like(means_mat, dtype=int)
for q in range(num_q):
    order = np.argsort(-means_mat[:, q])  # descending
    for rank, mi in enumerate(order, start=1):
        question_ranks[mi, q] = rank

# c) overall mean-of-means and its rank
first_mat = np.zeros((num_models, num_q), dtype=float)
for i, name in enumerate(model_names):
    df = model_dfs[name]
    # take only the 0th element of each is_corrects list
    first_mat[i, :] = np.array([lst[0] for lst in df["is_corrects"]])

# compute each model’s overall average of that first-sample across questions
overall_first_mean = first_mat.mean(axis=1)  # shape: (num_models,)

# rank models by descending first-sample mean (1 = best)
overall_order = np.argsort(-overall_first_mean)
overall_rank  = np.empty(num_models, dtype=int)
for rank, m_idx in enumerate(overall_order, start=1):
    overall_rank[m_idx] = rank

# d) combine into one matrix [models × (questions+1)]
combined = np.hstack([question_ranks, overall_rank[:, None]])

# e) plot
labels = [str(i+1) for i in range(num_q)] + ["avg"]
fig, ax = plt.subplots(figsize=(50, 3))
heat = ax.imshow(combined, aspect="auto", vmin=1, vmax=num_models, cmap="viridis")

# annotate each cell
for i in range(num_models):
    for j in range(num_q+1):
        ax.text(j, i, str(combined[i, j]),
                ha="center", va="center", color="white", fontsize=6)

ax.set_xticks(np.arange(num_q+1))
ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_yticks(np.arange(num_models))
ax.set_yticklabels(model_names, fontsize=8)

ax.set_xlabel(f"Question # (1–{num_q}) and overall", fontsize=10)
ax.set_ylabel("Models", fontsize=10)
cbar = fig.colorbar(heat, ax=ax, pad=0.02)
cbar.set_label("Rank (1 = best)", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("combined_ranks_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()