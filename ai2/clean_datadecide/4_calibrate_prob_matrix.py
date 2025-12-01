from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))
from utils import calibrate

REPO_ID = "yuhengtu/irsl_datadecide"
SNAPSHOT_DIR = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset"))
PROB_PATH = SNAPSHOT_DIR / "3_prob_matrix.parquet"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

prob_df = pd.read_parquet(PROB_PATH)

train_df = prob_df[prob_df.index.get_level_values("model_split") == "train"].copy()
test_df = prob_df[prob_df.index.get_level_values("model_split") == "test"].copy()

for name, df in [("train", train_df), ("test", test_df)]:
    total_cells = df.size
    nan_count = int(np.isnan(df.to_numpy()).sum())
    nan_pct = (nan_count / total_cells * 100) if total_cells else 0.0
    print(f"{name} rows: {len(df)}, questions: {df.shape[1]}")
    print(f"{name} NaN: {nan_count} cells ({nan_pct:.2f}%)")

probmat_train = torch.tensor(
    train_df.to_numpy(dtype=np.float32),
    device=device,
)
z_optimized = calibrate(probmat_train, device=device, batch_size=100)

mean_train = np.nanmean(train_df.to_numpy(), axis=0)
mean_test = np.nanmean(test_df.to_numpy(), axis=0)
rho_train, _ = spearmanr(z_optimized, mean_train)
rho_test, _ = spearmanr(z_optimized, mean_test)
print(
    f"\ncorr with np.nanmean(train_df[questions], axis=0) = {rho_train:.6f}"
    f"\ncorr with np.nanmean(test_df[questions], axis=0) = {rho_test:.6f}\n"
)
