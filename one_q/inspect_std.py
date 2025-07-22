import os
import json
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from tqdm import tqdm

# 1. Download all the JSON files from stair-lab/monkey_queries
# data_path = snapshot_download(repo_id="stair-lab/monkey_query_zero_shot", repo_type="dataset")
data_path = snapshot_download(repo_id="stair-lab/monkey_queries", repo_type="dataset")

# 2. Find every *_gsm.json file in the dataset folder
json_files = [fn for fn in os.listdir(data_path) if fn.endswith("_gsm.json")]

all_bootstrap_stds = []

# 3. For each model:
for fn in tqdm(sorted(json_files)):
    model_name = os.path.splitext(fn)[0]      # e.g. "Qwen3-8B_gsm"
    if model_name.startswith("Pythia") or model_name.startswith("Mistral"):
        continue
    
    with open(os.path.join(data_path, fn), "r") as f:
        entries = json.load(f)                # list of { "question": ..., "is_corrects": [...] }
    print(model_name, len(entries))

    # For each question in this model:
    for item in entries:
        samples = np.array(item["is_corrects"], dtype=float)[:2500]
        # Skip questions with no valid samples
        if len(samples) == 0 or np.all(np.isnan(samples)):
            continue

        # 100 bootstrap resamples of the samples (with replacement)
        boot_means = []
        for _ in range(100):
            resample = np.random.choice(samples, size=len(samples), replace=True)
            boot_means.append(np.nanmean(resample))
        # std of the bootstrap means
        std = np.nanstd(boot_means)
        all_bootstrap_stds.append(std)

# 4. Compute and print the average bootstrap‐estimated std
average_std = np.nanmean(all_bootstrap_stds)
print(f"Average bootstrap std across all models and questions: {average_std:.4f}")
