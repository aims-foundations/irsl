from huggingface_hub import HfApi, login
import glob
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

BENCHMARK2SCENARIO = {
    "lite": ["legalbench", "commonsense", "med_qa"], # "math", "gsm"
    "mmlu": ["mmlu"],
    "classic": ["bbq", "lsat_qa", "legal_support"],
}
SCENARIO2BENCHMARK = {
    **{
        scen: bench
        for bench, scens in BENCHMARK2SCENARIO.items()
        for scen in scens
    },
}
SCENARIOS = sorted(SCENARIO2BENCHMARK.keys())

if __name__ == "__main__":
    cache_dir = snapshot_download(repo_id="stair-lab/monkey_queries", repo_type="dataset")
    all_paths = []
    for scen in SCENARIOS:
        all_paths.extend(glob.glob(f"{cache_dir}/*{scen}.json"))

    # build union of all model names
    model_set = set()
    for path in all_paths:
        stem = Path(path).stem
        scen = next((s for s in SCENARIOS if stem.endswith(f"_{s}")), None)
        model = stem.split(f"_{scen}.json")[0]
        model_set.add(model)
    model_names = sorted(model_set)
    n_models = len(model_names)
    print(n_models, model_names)

    # build union of all (benchmark, scenario, prompt) pairs
    qpairs = set()
    max_samples = 0
    for path in all_paths:
        stem = Path(path).stem
        scen = next((s for s in SCENARIOS if stem.endswith(f"_{s}")), None)
        bench = SCENARIO2BENCHMARK[scen]
        df_monkey = pd.read_json(path)
        qs = df_monkey["question"].tolist()
        for q in qs:
            qpairs.add((bench, scen, q))
        max_samples = max(max_samples, len(df_monkey["is_corrects"].tolist()))
    qpairs = sorted(qpairs)
    
    # load z from helm_resmat
    with open("/lfs/skampere1/0/sttruong/deval/data/gather_helm_data/results_with_z.pkl", "rb") as f:
        helm_resmat = pickle.load(f)
    cols = helm_resmat.columns
    helm_benchs = cols.get_level_values("benchmark")
    helm_scens = cols.get_level_values("scenario")
    helm_qs = cols.get_level_values("input.text").astype(str)
    helm_refs = cols.get_level_values("references").astype(str)
    helm_zs = cols.get_level_values("z")
    qpairs_with_zs = set()
    for bench, scen, q in qpairs:
        mask = (helm_benchs == bench) & (helm_scens == scen)
        if scen == "legal_support":
            mask &= (helm_qs + helm_refs == q)
        else:
            mask &= (helm_qs == q)
        matched_z = helm_zs[mask]
        if len(matched_z) >= 1:
            qpairs_with_zs.add((bench, scen, q, matched_z.mean()))
        else:
            continue
    qpairs_with_zs = sorted(qpairs_with_zs)
    n_questions = len(qpairs_with_zs)
    print(n_questions)
    
    # allocate 3D array filled with NaN
    data_tensor = np.full(
        (n_models, n_questions, max_samples),
        np.nan,
        dtype=float
    )

    # fill in scores
    output_benchs, output_scens, output_qs, output_zs = [], [], [], []
    for i, m in tqdm(enumerate(model_names)):
        for j, (bench, scen, q, z) in enumerate(qpairs_with_zs):
            path = f"{cache_dir}/{m}_{scen}.json"
            df_monkey = pd.read_json(path)
            scores = df_monkey.loc[df_monkey["question"] == q, "is_corrects"].to_numpy()
            L = len(scores)
            if L > 0:
                data_tensor[i, j, :L] = scores
            output_benchs.append(bench)
            output_scens.append(scen)
            output_qs.append(q)
            output_zs.append(z)
    
    print(f"shape {data_tensor.shape}")
    total_elements = np.prod(data_tensor.shape)
    nan_count = np.isnan(data_tensor).sum()
    nan_percentage = (nan_count / total_elements) * 100
    print(f"NaN percentage: {nan_percentage:.2f}%, NaN count: {nan_count}")
    
    scenarios_unique = sorted(set(output_scens))
    for scen in scenarios_unique:
        idxs = [j for j, s in enumerate(output_scens) if s == scen]
        print(f"{scen}: shape = (n_models={n_models}, n_questions={len(idxs)}, max_samples={max_samples})")
        
    # save to disk
    out_path = Path("../../data/testtime_resmat1.pt")
    torch.save({
        "data_tensor": torch.from_numpy(data_tensor),
        "models": model_names,
        "benchmarks": output_benchs,
        "scenarios": output_scens,
        "questions": output_qs,
        "zs": output_zs
    }, out_path)

    login()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo="testtime_resmat1.pt",
        repo_id="stair-lab/irsl_testtime_resmat1",
        repo_type="dataset",
    )