import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from scipy.special import logit
from tqdm import tqdm

# ─── Step 1: Load GSM8K “platinum” questions ───────────────────────────────────
platinum = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
valid = platinum.filter(lambda ex: ex["cleaning_status"] in ["consensus", "verified"])
platinum_questions = set(valid["question"])

# ─── Step 2: Pull down your HF dataset of JSON outputs ─────────────────────────
# repo_dir = Path(snapshot_download(
#     repo_id="stair-lab/one_question_less_samples",
#     repo_type="dataset"
# ))
# json_files = sorted(repo_dir.glob("*.json"))

repo_dir = Path(snapshot_download(
    repo_id="stair-lab/monkey_queries",
    repo_type="dataset"
))
selected_files = [
    "DeepSeek-R1-Distill-Llama-8B_gsm.json",
    "DeepSeek-V2-Lite-Chat_gsm.json",
    "Mistral-7B-v0.1_gsm.json",
    "Qwen3-14B_gsm.json",
    "Qwen3-32B_gsm.json",
    "Qwen3-8B_gsm.json",
    "gemma-3-27b-it_gsm.json",
]
json_files = [repo_dir / fname for fname in selected_files]


# ─── Step 3: Read each model’s JSON into a DataFrame ──────────────────────────
model_dfs = {}
for jf in json_files:
    with open(jf, "r", encoding="utf-8") as f:
        records = json.load(f)
    model_name = jf.stem.split("_")[0]         # assumes names like "ModelX_gsm.json"
    df = pd.DataFrame(records)                 # columns: ["question", "is_corrects"]
    model_dfs[model_name] = df

# ─── Step 4: Find the common questions across all models ──────────────────────
all_sets = [set(df["question"]) for df in model_dfs.values()]
common_qs = set.intersection(*all_sets)

# ─── Step 5: Restrict to platinum questions ────────────────────────────────────
common_plat = common_qs & platinum_questions

# ─── Step 6: Filter each DataFrame to only those questions ────────────────────
for name, df in model_dfs.items():
    df = df[df["question"].isin(common_plat)].copy()
    df.reset_index(drop=True, inplace=True)
    model_dfs[name] = df

# ─── Step 7: Compute per-question mean and first-sample matrices ───────────────
model_names = list(model_dfs.keys())
T = len(model_names)
Q = len(common_plat)

means_mat = np.zeros((T, Q), dtype=float)
first_mat = np.zeros((T, Q), dtype=float)

for i, name in enumerate(model_names):
    is_corr_lists = model_dfs[name]["is_corrects"].tolist()  # list of Q lists of length 50
    # (a) mean over all 50 samples
    means_mat[i, :] = [np.nanmean(lst) for lst in is_corr_lists]
    # (b) first‐sample only
    first_mat[i, :] = [lst[0] for lst in is_corr_lists]

# overall average of first-sample across questions
overall_first_mean = first_mat.mean(axis=1)  # shape (T,)

# ─── Step 8: Build the probability and logit arrays ───────────────────────────
probs = np.hstack([means_mat, overall_first_mean[:, None]])  # shape (T, Q+1)
eps = 1e-6
probs = np.clip(probs, eps, 1 - eps)
logits = logit(probs)  # shape (T, Q+1)

# ─── Step 9: Reconstruct the raw lists for bootstrap ──────────────────────────
# probs_raw[i][j] is the list of 50 floats for model i, question j;
# plus the “first-sample” list for the avg column.
probs_raw = []
for name in model_names:
    lists = model_dfs[name]["is_corrects"].tolist()  # Q lists
    first_sample = [lst[0] for lst in lists]
    lists.append(first_sample)                      # now length Q+1
    probs_raw.append(lists)

# ─── Step 10: Bootstrap to estimate logit‐std at each (model, question) ───────
n_boot = 100
frac   = 0.8
stds = np.zeros_like(logits)

for i in tqdm(range(T), desc="Bootstrapping"):
    for j in range(Q+1):
        data = np.array(probs_raw[i][j])
        m = int(np.ceil(len(data) * frac))
        boot = np.zeros(n_boot, dtype=float)
        for b in range(n_boot):
            sample = np.random.choice(data, size=m, replace=True)
            p_hat = sample.mean()
            p_hat = np.clip(p_hat, eps, 1 - eps)
            boot[b] = logit(p_hat)
        stds[i, j] = boot.std(ddof=1)

# ─── Step 11: Plot with vertical stems, errorbars, and dashed lines ──────────
x = np.arange(Q+1)
x_labels = [str(k+1) for k in range(Q)] + ["avg"]
cmap = plt.get_cmap("tab10", T)
colors = [cmap(i) for i in range(T)]

fig, ax = plt.subplots(figsize=(max(6, (Q+1)*0.4), 6))

# # 11a. gray vertical stems
# for j in range(Q+1):
#     yj = logits[:, j]
#     ax.vlines(x[j], ymin=yj.min(), ymax=yj.max(), color="gray", alpha=0.4)

# 11b. dashed lines + errorbars per model
for i, name in enumerate(model_names):
    ax.errorbar(
        x, logits[i], yerr=stds[i],
        linestyle="--", marker="o", markersize=5,
        color=colors[i], elinewidth=1, capsize=3,
        label=name
    )

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=90)
ax.set_xlabel("Question # and overall ('avg')")
ax.set_ylabel("Logit (inverse‐sigmoid) of Probability")
ax.set_title("Per‐Question & Overall Logits Across Models\n(±1 boot‐std, 80% samples × 100 resamples)")

# legend outside
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

plt.tight_layout()
fig.savefig("logits_with_bootstrap_std.png", dpi=300, bbox_inches="tight")
plt.show()
