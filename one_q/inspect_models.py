import os
import json
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

# 1. Download all the JSON files from stair-lab/monkey_queries
data_path = snapshot_download(repo_id="stair-lab/monkey_queries", repo_type="dataset")

# 2. Find every *_gsm.json file in the dataset folder
json_files = [fn for fn in os.listdir(data_path) if fn.endswith("_gsm.json")]

models = []
avg_scores_per_model = []

# 3. Load each JSON, extract & average the 'is_corrects' list per question
for fn in sorted(json_files):
    model_name = os.path.splitext(fn)[0]      # e.g. "Qwen3-8B_gsm"
    if model_name.startswith("Pythia") or model_name.startswith("Mistral"):
        continue
    models.append(model_name)
    with open(os.path.join(data_path, fn), "r") as f:
        entries = json.load(f)                # list of { "question": ..., "is_corrects": [...] }
    print(model_name, len(entries))
    # compute mean correctness per question
    means = [np.nanmean(item["is_corrects"]) for item in entries]
    avg_scores_per_model.append(means)

M = len(models)

# 4. Plot one histogram per model in a single‑column figure
fig, axes = plt.subplots(M, 1, figsize=(6, M * 3), constrained_layout=True)
axes = axes.flatten()

for i, (model, scores) in enumerate(zip(models, avg_scores_per_model)):
    ax = axes[i]
    ax.hist(scores, bins=20, edgecolor="black")
    ax.set_title(model, fontsize=10)
    ax.set_xlabel("Avg correctness per question")
    ax.set_ylabel("Number of questions")
    ax.set_xlim(0, 1)

# 5. Save the figure
plt.savefig("model_histograms.png", dpi=300, bbox_inches="tight")
plt.close(fig)
