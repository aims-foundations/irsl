from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent / "data"
INPUT_PATH = BASE_DIR / "2_datadecide_long.parquet"
OUTPUT_BINARY = BASE_DIR / "3_binary_matrix.parquet"
OUTPUT_PROB = BASE_DIR / "3_prob_matrix.parquet"
OUTPUT_BPB = BASE_DIR / "3_bpb_matrix.parquet"
FIG_DIR = Path(__file__).resolve().parent / "results" / "3_clean_and_pivot"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PROB_THRESHOLD = 0.15

frame = pd.read_parquet(INPUT_PATH)

### 1. clean
print("### 1. clean")

# Aggregate across seeds
print("Aggregating across seeds")
seed_group_cols = ["model_data_mix", "model_size", "model_step", "bench_name", "doc_id", "correct_choice"]
choice_logit_cols = [col for col in frame.columns if col.startswith("choice_")]

def majority_binary(series: pd.Series) -> int:
    return int(np.nanmean(series) > 0.5)
agg_spec = {"acc_per_char": majority_binary}
for col in choice_logit_cols:
    agg_spec[col] = "mean"  # mean skips nan by default

frame = frame.groupby(seed_group_cols, as_index=False).agg(agg_spec)

# Split model_data_mix values into deterministic train/test buckets
mixes = frame["model_data_mix"].unique().tolist()
random.Random(0).shuffle(mixes)
train_mixes, test_mixes = mixes[:5],mixes[5:]

def assign_split(mix: str) -> str:
    return "train" if mix in train_mixes else "test"

frame["model_split"] = frame["model_data_mix"].map(assign_split)

# Extract correct_bqb and p_correct_choice
correct_choices = frame["correct_choice"].to_numpy(dtype=int)

bpb_cols = sorted(
    [col for col in frame.columns if col.startswith("choice_") and col.endswith("_logits_per_byte")],
    key=lambda c: int(c.split("_")[1]),
)
bpb_values = frame[bpb_cols].to_numpy()
frame["correct_bpb"] = bpb_values[np.arange(len(frame)), correct_choices].astype(float)

prob_cols = sorted(
    [col for col in frame.columns if col.startswith("choice_") and col.endswith("_logits_per_char")],
    key=lambda c: int(c.split("_")[1]),
)
prob_values = frame[prob_cols].to_numpy()
frame["p_correct_choice"] = np.exp(prob_values[np.arange(len(frame)), correct_choices].astype(float))

### 2. pivot
print("### 2. pivot")
row_indices = ["model_data_mix", "model_size", "model_step", "model_split"]
col_indices = ["bench_name", "doc_id"]

binary_matrix = frame.pivot_table(
    index=row_indices,
    columns=col_indices,
    values="acc_per_char",
    aggfunc="first",
)
prob_matrix = frame.pivot_table(
    index=row_indices,
    columns=col_indices,
    values="p_correct_choice",
    aggfunc="mean",
)
bpb_matrix = frame.pivot_table(
    index=row_indices,
    columns=col_indices,
    values="correct_bpb",
    aggfunc="mean",
)

# filling nan for missed rows/cols
prob_matrix = prob_matrix.reindex(index=binary_matrix.index, columns=binary_matrix.columns)
bpb_matrix = bpb_matrix.reindex(index=binary_matrix.index, columns=binary_matrix.columns)
print(f"binary_matrix shape: {binary_matrix.shape}, prob_matrix shape: {prob_matrix.shape}, bpb_matrix shape: {bpb_matrix.shape}")
assert binary_matrix.shape == prob_matrix.shape == bpb_matrix.shape

### 3. filter out low score test takers
print("### 3. filter out low score test takers")
prob_row_mean = prob_matrix.mean(axis=1, skipna=True)
binary_row_mean = binary_matrix.mean(axis=1, skipna=True)
bpb_row_mean = bpb_matrix.mean(axis=1, skipna=True)
for name, row_mean in [
    ("prob", prob_row_mean),
    ("binary", binary_row_mean),
    ("bpb", bpb_row_mean),
]:
    plt.figure()
    plt.hist(row_mean.dropna(), bins=50, density=True)
    plt.savefig(FIG_DIR / f"rowavg_distri_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

rows_to_drop = set(prob_row_mean[prob_row_mean < PROB_THRESHOLD].index)
print(rows_to_drop)
binary_matrix = binary_matrix.drop(index=rows_to_drop)
prob_matrix = prob_matrix.drop(index=rows_to_drop)
bpb_matrix = bpb_matrix.drop(index=rows_to_drop)
print(f"Dropped {len(rows_to_drop)} low-average rows; remaining rows: {len(binary_matrix)}")

for name, matrix in [
    ("binary_matrix", binary_matrix),
    ("prob_matrix", prob_matrix),
    ("bpb_matrix", bpb_matrix),
]:
    total_cells = matrix.size
    nan_count = int(matrix.isna().sum().sum())
    nan_pct = (nan_count / total_cells * 100)
    print(f"{name} shape: {matrix.shape}")
    print(f"{name} NaN: {nan_count} cells ({nan_pct:.2f}%)")

binary_matrix.to_parquet(OUTPUT_BINARY)
prob_matrix.to_parquet(OUTPUT_PROB)
bpb_matrix.to_parquet(OUTPUT_BPB)
