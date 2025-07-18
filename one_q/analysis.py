import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download

# 1. Load GSM8K platinum and select only valid questions
platinum_dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
filtered_platinum = platinum_dataset.filter(lambda item: item["cleaning_status"] in ["consensus", "verified"])
platinum_question_set = set(filtered_platinum["question"])

# 2. Retrieve and load the model response tensor
dataset_path = snapshot_download(repo_id="stair-lab/monkey_3d_data", repo_type="dataset")
tensor_file = os.path.join(dataset_path, "gsm_tensor.pth")
monkey_data = torch.load(tensor_file, map_location="cpu")

raw_questions   = monkey_data["questions"]      # list[str]
all_models      = monkey_data["models"]         # list[str]
score_tensor    = monkey_data["data_tensor"]    # shape: [M, Q, S]

# 3. Exclude specific models
exclude_list   = [
    "Meta-Llama-3-8B-Instruct", "Pythia_6.9B", "Meta-Llama-3-70B-Instruct",
    "gemma-3-27b-it", "Mistral-7B-v0.1", "Pythia_12B",
]
model_mask        = [name not in exclude_list for name in all_models]
filtered_scores   = score_tensor[model_mask, :, :]
remaining_models  = [name for name in all_models if name not in exclude_list]

# 4. Keep only questions present in the platinum set
question_mask = [q in platinum_question_set for q in raw_questions]
filtered_scores = filtered_scores[:, question_mask, :]
raw_questions    = [q for q in raw_questions if q in platinum_question_set]

# 5. Remove any slices that are all NaN
#    - drop questions with all NaN across models & samples
retain_q_mask   = ~torch.all(torch.isnan(filtered_scores), dim=(0, 2))
filtered_scores = filtered_scores[:, retain_q_mask, :]
raw_questions   = [raw_questions[i] for i, keep in enumerate(retain_q_mask) if keep]

#    - drop samples with all NaN across models & questions
retain_s_mask   = ~torch.all(torch.isnan(filtered_scores), dim=(0, 1))
filtered_scores = filtered_scores[:, :, retain_s_mask]
print("Tensor shape after NaN cleanup:", filtered_scores.shape)

# 6. Compute mean score per question and assign ranks (1 = best)
mean_scores     = filtered_scores.mean(axis=2)  # [models, questions]
question_ranks  = np.zeros_like(mean_scores, dtype=int)
for q_idx in range(mean_scores.shape[1]):
    sorted_idx = np.argsort(-mean_scores[:, q_idx])
    for rank, m_idx in enumerate(sorted_idx, start=1):
        question_ranks[m_idx, q_idx] = rank

# 7. Load HELM benchmark results
helm_results = pd.read_pickle("gsm_results.pkl")

# 8. Select only columns matching platinum questions
valid_columns = helm_results.columns.get_level_values("input.text").isin(platinum_question_set)
helm_results  = helm_results.loc[:, valid_columns]
breakpoint()

# 9. Map full model names to short names and align with retained models
helm_results = helm_results.reset_index().rename(columns={"request.model": "full_model"})
helm_results["short_model"] = helm_results["full_model"].str.split("/").str[-1]
helm_results = (helm_results
                .set_index("short_model")
                .reindex(remaining_models)
                .drop(columns=["full_model"]))
breakpoint()

# 10. Compute average HELM rank (descending mean → rank 1)
helm_mean_scores = helm_results.values.mean(axis=1)
average_ranks    = np.argsort(-helm_mean_scores) + 1

# 11. Combine per‑question ranks with overall rank and visualize
combined_ranks = np.hstack([question_ranks, average_ranks[:, None]])
column_labels = [str(i + 1) for i in range(question_ranks.shape[1])] + ["avg"]

fig, ax = plt.subplots(figsize=(20, 5))
heatmap = ax.imshow(combined_ranks, aspect="auto", vmin=1, vmax=len(remaining_models))

for row in range(combined_ranks.shape[0]):
    for col in range(combined_ranks.shape[1]):
        ax.text(col, row, str(combined_ranks[row, col]), ha="center", va="center", fontsize=6)

ax.set_xticks(np.arange(len(column_labels)))
ax.set_xticklabels(column_labels, rotation=90, fontsize=6)
ax.set_yticks(np.arange(len(remaining_models)))
ax.set_yticklabels(remaining_models, fontsize=8)
ax.set_xlabel(f"Question # (1–{question_ranks.shape[1]}) & average", fontsize=10)
ax.set_ylabel("Models", fontsize=10)

cbar = fig.colorbar(heatmap, ax=ax, pad=0.02)
cbar.set_label("Rank (1 = best)", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("ranking_table.png", dpi=300, bbox_inches="tight")