from pathlib import Path
import numpy as np
import pandas as pd
rng = np.random.default_rng(0)

BASE_DIR = Path(__file__).resolve().parent / "data"
BUDGET = 50

SEQUENCE_LENGTH = 2048
MODEL2BATCH = {
    '4M': 32, # batch_size=32, gpus=8
    '6M': 32,
    '8M': 32,
    '10M': 32,
    '14M': 32,
    '16M': 32,
    '20M': 64,
    '60M': 96,
    '90M': 160,
    '150M': 192,
    '300M': 320,
    '530M': 448,
    '750M': 576,
    '1B': 704
}
MODEL2PARA = {
    '4M': 3_744_832,
    '6M': 6_010_464,
    '8M': 8_538_240,
    '10M': 9_900_432,
    '12M': 12_066_600,
    '14M': 14_380_224,
    '16M': 16_004_560,
    '20M': 19_101_888,
    '60M': 57_078_144,
    '90M': 97_946_640,
    '150M': 151_898_880,
    '300M': 319_980_544,
    '530M': 530_074_944,
    '750M': 681_297_408,
    '1B': 1_176_832_000
}
def calculate_flops(
    model_size: str,
    step: int,
) -> float:
    n = MODEL2PARA[model_size]
    d = MODEL2BATCH[model_size] * step * SEQUENCE_LENGTH
    return float(n * d * 6)

INPUT_BINARY = BASE_DIR / "5_binary_matrix_cated.parquet"
INPUT_PROB = BASE_DIR / "5_prob_matrix_cated.parquet"
INPUT_BPB = BASE_DIR / "3_bpb_matrix.parquet"
OUTPUT_PATH = BASE_DIR / "6_organized.parquet"

binary_df = pd.read_parquet(INPUT_BINARY)
binary_test_df = binary_df[binary_df.index.get_level_values("model_split") == "test"].copy()
prob_df = pd.read_parquet(INPUT_PROB)
prob_test_df = prob_df[prob_df.index.get_level_values("model_split") == "test"].copy()
bpb_df = pd.read_parquet(INPUT_BPB)
bpb_test_df = bpb_df[bpb_df.index.get_level_values("model_split") == "test"].copy()

question_index_names = ["bench_name", "doc_id"]
binary_question_index = binary_test_df.columns.to_frame(index=False)[question_index_names]
prob_question_index = prob_test_df.columns.to_frame(index=False)[question_index_names]
bpb_question_index = bpb_test_df.columns.to_frame(index=False)[question_index_names]
assert binary_question_index.equals(prob_question_index) \
    and binary_question_index.equals(bpb_question_index)

binary_base_index = binary_test_df.index.to_frame(index=False)
prob_base_index = prob_test_df.index.to_frame(index=False)
bpb_base_index = bpb_test_df.index.to_frame(index=False)

model_index_names = ["model_data_mix", "model_size", "model_step"]
assert binary_base_index[model_index_names].equals(prob_base_index[model_index_names]) \
    and binary_base_index[model_index_names].equals(bpb_base_index[model_index_names])

model_index = binary_base_index[model_index_names]
max_model_step = (
    model_index.groupby(["model_data_mix", "model_size"], as_index=False)["model_step"]
    .max()
    .rename(columns={"model_step": "max_model_step"})
)
model_index = model_index.merge(max_model_step, on=["model_data_mix", "model_size"], how="left")
final_step_mask = model_index["model_step"] == model_index["max_model_step"]
model_index["FLOP"] = [np.nan] * len(model_index)
model_index.loc[final_step_mask, "FLOP"] = [
    calculate_flops(model_size, step)
    for model_size, step in zip(
        model_index.loc[final_step_mask, "model_size"], model_index.loc[final_step_mask, "model_step"]
    )
]
assert model_index[model_index_names].equals(binary_base_index[model_index_names])

long_df = model_index[["model_data_mix", "model_size", "model_step", "FLOP"]].copy()
ability_index_names = [c for c in binary_base_index.columns if c.startswith("ability_")]
for ability_index_name in ability_index_names:
    long_df[f"{ability_index_name}_binary"] = binary_base_index[ability_index_name]
    long_df[f"{ability_index_name}_prob"] = prob_base_index[ability_index_name]

bench_names = binary_test_df.columns.get_level_values("bench_name").map(
    lambda b: "mmlu" if b.startswith("mmlu") else b
)
unique_bench_names = sorted(bench_names.unique())
binary_zs = binary_test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
beta_zs = prob_test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
binary_ys = binary_test_df.to_numpy(dtype=np.float32)
beta_ys = prob_test_df.to_numpy(dtype=np.float32)
bpb_ys = bpb_test_df.to_numpy(dtype=np.float32)

for bench in unique_bench_names:
    bench_mask = bench_names == bench
    bench_binary_ys = binary_ys[:, bench_mask]
    bench_beta_ys = beta_ys[:, bench_mask]
    bench_bpb_ys = bpb_ys[:, bench_mask]
    bench_binary_zs = binary_zs[bench_mask]
    bench_beta_zs = beta_zs[bench_mask]
    sampled_idx = rng.choice(len(bench_binary_ys.shape[-1]), size=BUDGET, replace=False)

    long_df[f"difficulties_{bench}_binary"] = [bench_binary_zs.tolist()] * len(long_df)
    long_df[f"difficulties_{bench}_beta"] = [bench_beta_zs.tolist()] * len(long_df)
    long_df[f"acc_full_{bench}"] = bench_binary_ys.mean(axis=1, skipna=True)
    long_df[f"acc_sub_{bench}"] = bench_binary_ys[:, sampled_idx].mean(axis=1, skipna=True)
    long_df[f"p_correct_choice_full_{bench}"] = bench_beta_ys.mean(axis=1, skipna=True)
    long_df[f"p_correct_choice_sub_{bench}"] = bench_beta_ys[:, sampled_idx].mean(axis=1, skipna=True)
    long_df[f"correct_bpb_full_{bench}"] = bench_bpb_ys.mean(axis=1, skipna=True)
    long_df[f"correct_bpb_sub_{bench}"] = bench_bpb_ys[:, sampled_idx].mean(axis=1, skipna=True)

long_df.to_parquet(OUTPUT_PATH)