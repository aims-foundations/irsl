import os
import glob
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

# 1) Define your ordered scenarios and mappings
ordered_scenarios = ["gsm"] #, "mmlu", "lsat_qa", "legalbench",
    # "bbq", "commonsense", "math", "med_qa", "legal_support"]

benchmark2scenario = {
    "lite":    ["gsm"],
    # "lite":    ["legalbench", "math", "commonsense", "med_qa", "gsm"],
    # "mmlu":    ["mmlu"],
    # "classic": ["bbq", "lsat_qa", "legal_support"]
}
scenario2benchmark = {
    **{
        scen: bench
        for bench, scenarios in benchmark2scenario.items()
        for scen in scenarios
    },
    # 'harm_bench': 'safety',
}

# 2) Download all monkey query JSONs
local_dir = snapshot_download(
    repo_id="stair-lab/monkey_query_zero_shot",
    repo_type="dataset"
)
all_paths = []
for scen in scenario2benchmark:
    all_paths.extend(glob.glob(f"{local_dir}/*{scen}.json"))

# 3) Organize paths by scenario
paths_by_scenario = defaultdict(list)
for path in all_paths:
    stem = Path(path).stem
    scen = next((s for s in scenario2benchmark if stem.endswith(f"_{s}")), None)
    if scen is None or scen == "harm_bench":
        continue
    raw_model = stem[: -(len(scen) + 1)]
    # skip unwanted Pythia
    if raw_model.startswith("Pythia"):
        continue
    paths_by_scenario[scen].append((raw_model, path))

# helper: load & filter HELM DataFrame
def load_and_filter_helm(benchmark_name, scenario_name, pkl_dir):
    pkl_fname = "results_with_z" if benchmark_name != "safety" else "results_with_z_harmbench"
    with open(os.path.join(pkl_dir, f"{pkl_fname}.pkl"), "rb") as f:
        helm_resmat = pickle.load(f)
    # drop nan z
    helm_resmat = helm_resmat.loc[:, ~helm_resmat.columns.get_level_values("z").isna()]
    # filter benchmark & scenario
    helm_resmat = helm_resmat.loc[:,
        (helm_resmat.columns.get_level_values("benchmark") == benchmark_name) &
        (helm_resmat.columns.get_level_values("scenario")  == scenario_name)
    ]
    helm_resmat = helm_resmat.dropna(how="all", axis=0)
    if scenario_name == "legal_support":
        cols = helm_resmat.columns.to_frame(index=False)
        cols["input.text"] = cols["input.text"].astype(str) + cols["references"].astype(str)
        helm_resmat.columns = pd.MultiIndex.from_frame(cols)
    # drop duplicate questions
    helm_resmat = helm_resmat.loc[:, ~helm_resmat.columns.get_level_values("input.text").duplicated(keep=False)]
    return helm_resmat

# 4) Process each scenario
output = {}
device = torch.device("cpu")
helm_dir = "/lfs/skampere1/0/sttruong/deval/data/gather_helm_data"

for scen in ordered_scenarios:
    entries = paths_by_scenario.get(scen, [])
    if not entries:
        continue

    bench = scenario2benchmark[scen]
    # 4.1) First pass: gather each model's question set
    question_sets = []
    for model_name, path in entries:
        df_monkey = pd.read_json(path)
        q2correct = {row["question"]: row["is_corrects"] for _, row in df_monkey.iterrows()}

        helm_resmat = load_and_filter_helm(bench, scen, helm_dir)
        helm_resmat = helm_resmat.loc[:, helm_resmat.columns.get_level_values("input.text").isin(q2correct)]
        questions = helm_resmat.columns.get_level_values("input.text").tolist()
        question_sets.append(set(questions))

    common_questions = set.union(*question_sets)
    question_texts = sorted(common_questions)

    # extract z values from any one model's HELM after filter
    helm_sample = load_and_filter_helm(bench, scen, helm_dir)
    helm_sample = helm_sample.loc[:, helm_sample.columns.get_level_values("input.text").isin(question_texts)]
    z_vals = helm_sample.columns.get_level_values("z").astype(float).to_numpy()
    z_per_question = torch.tensor(z_vals, dtype=torch.float64, device=device)

    # 4.2) Second pass: build per-model arrays
    per_model_lists = []
    model_list = []
    global_max_samples = 0

    for model_name, path in entries:
        df_monkey = pd.read_json(path)
        q2correct = {row["question"]: row["is_corrects"] for _, row in df_monkey.iterrows()}

        this_model_arr = []
        for q in question_texts:
            arr = np.array(q2correct.get(q, []), dtype=float)
            this_model_arr.append(arr)
            global_max_samples = max(global_max_samples, arr.size)

        per_model_lists.append(this_model_arr)
        model_list.append(model_name)

    # 4.3) Allocate and fill tensor
    n_models = len(model_list)
    n_questions = len(question_texts)
    data_tensor = torch.full(
        (n_models, n_questions, global_max_samples),
        float('nan'),
        dtype=torch.float64,
        device=device
    )
    for i, model_arr in enumerate(per_model_lists):
        for j, arr in enumerate(model_arr):
            data_tensor[i, j, : arr.size] = torch.from_numpy(arr)

    output[scen] = {
        "data_tensor": data_tensor,
        "z":           z_per_question,
        "models":      model_list,
        "questions":   question_texts,
    }

# 5) Save each scenario's results
out_dir = "../../data/monkey_3d_tensor_zero_shot"
os.makedirs(out_dir, exist_ok=True)
for scen, info in output.items():
    save_path = os.path.join(out_dir, f"{scen}_tensor.pth")
    torch.save(info, save_path)
    print(f"Saved {scen}: {info['data_tensor'].shape} with {len(info['models'])} models, {len(info['questions'])} questions, {info['data_tensor'].shape[-1]} samples.")
