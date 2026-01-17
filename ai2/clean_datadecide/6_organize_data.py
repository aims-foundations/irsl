from itertools import islice
from pathlib import Path

import pandas as pd

# Inspect the prob matrix parquet with theta adjustments, focusing on index structure.
parquet_path = Path(__file__).with_name("5_prob_matrix_with_theta.parquet")
df = pd.read_parquet(parquet_path)

print(f"Shape: {df.shape}")

idx = df.index
print("\nRow index names:", idx.names)
print("Row index length:", len(idx))
print("Row index sample (first 5):", list(islice(idx, 5)))

cols = df.columns
print("\nColumn index name:", getattr(cols, "name", None))
print("Column index length:", len(cols))
print("Column index sample (first 5):", list(islice(cols, 5)))


# Tag model_is_final_step for each (model_data_mix, model_size)
group_cols = ["model_data_mix", "model_size"]
frame["model_step_numeric"] = pd.to_numeric(frame["model_step"], errors="coerce")
max_steps = (
    frame[group_cols + ["model_step_numeric"]]
        .groupby(group_cols, as_index=False)
        .max()
        .rename(columns={"model_step_numeric": "max_model_step"})
)
frame = frame.merge(max_steps, on=group_cols, how="left")
frame["model_is_final_step"] = frame["model_step_numeric"] == frame["max_model_step"]
