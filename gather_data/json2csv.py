import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import exists

lo = lambda x: json.load(open(x, "r"))

def infer_column_types(df):
    for col in df.columns:
        try:
            unique_values = df[col].dropna().unique()
        except:
            df[col] = df[col].apply(lambda x: json.dumps(x))
            unique_values = df[col].dropna().unique()
        
        if set(unique_values).issubset({"True", "False", "0", "1"}):
            df[col] = df[col].map(lambda x: True if x in ["True", "1"] else False).astype("bool")
        elif np.all(~pd.isna(pd.to_numeric(unique_values, errors="coerce"))):
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")
        elif df[col].nunique() / len(df) < 0.1:
            df[col] = df[col].astype("string").astype("category")

if __name__ == "__main__":
    BENCHMARKS = ["classic", "mmlu", "lite"]
    benchmark_dir = "/lfs/skampere1/0/yuhengtu/helm/src/benchmark_output/runs"

    task2metric = lo("task2metric.json")
    task2metric = pd.json_normalize(task2metric)

    all_paths = []
    for benchmark in BENCHMARKS:
        dirs = [d for d in os.listdir(benchmark_dir) if d.startswith(benchmark)]
        for d in dirs:
            full_d_path = f"{benchmark_dir}/{d}"
            if os.path.isdir(full_d_path):
                subdirs = [
                    sub for sub in os.listdir(full_d_path) if os.path.isdir(f"{full_d_path}/{sub}")
                ]
                all_paths.extend([f"{full_d_path}/{sub}" for sub in subdirs])
    files = ["display_requests.json", "display_predictions.json", "run_spec.json"]
    all_paths = [path for path in all_paths if not os.path.basename(path).endswith("step111000")]
    all_paths = [p for p in tqdm(all_paths) if all([exists(f"{p}/{f}") for f in files])]
    all_lists = [[lo(f"{p}/{f}") for p in tqdm(all_paths)] for f in files]

    results = []
    for d_requests, d_predictions, run_specs, paths in tqdm(zip(*all_lists, all_paths), total=len(all_lists[0])):
        d_requests = pd.json_normalize(d_requests)
        d_predictions = pd.json_normalize(d_predictions)
        run_specs = pd.json_normalize(run_specs)
        benchmark = paths.split("/")[-2].split("_")[0]
        n_step = paths.split("/")[-2].split("-")[-1]
        run_specs["benchmark"] = benchmark
        run_specs = run_specs.loc[run_specs.index.repeat(d_predictions.shape[0])].reset_index(drop=True)
        overlap_column = d_predictions.columns.intersection(d_requests.columns)
        d_requests = d_requests.drop(columns=overlap_column)
        result = pd.concat([d_requests, d_predictions, run_specs], axis=1)
        result["request.model"] = result["request.model"] + "-" + n_step
        result["scenario"] = result['name'].str.split(r'[:,]', n=1, expand=True)[0]
        result["scenario"] = result["scenario"].astype("category")
        assert result["scenario"].nunique() == 1
        metric_name = task2metric[f"{benchmark}.{result['scenario'].iloc[0]}"].iloc[0]
        if isinstance(metric_name, list):
            for metric_name_ in metric_name:
                dicho_score = result.get(f"stats.{metric_name_}", pd.NA)
                if dicho_score is not pd.NA:
                    if not dicho_score.isna().all():
                        result["dicho_score"] = dicho_score
                        break
        else:
            result["dicho_score"] = result.get(f"stats.{metric_name}", pd.NA)
        results.append(result)

    results = pd.concat(results, axis=0, join='outer')
    print("finished create results dataframe")
    infer_column_types(results)
    results.reset_index(drop=True, inplace=True)
    for col in results.columns:
        # if results[col].dtype != "category" and np.isnan(results[col]).all():
        if results[col].dtype != "category" and results[col].isna().all():
            results = results.drop(columns=col)
        else:
            if results[col].dtype == "float64" and np.nanmax(results[col]) < 65500 and np.nanmin(results[col]) > -65500:
                results[col] = results[col].astype("float16")
    print("Started saving results")
    output_dir = "../data/"
    os.makedirs(output_dir, exist_ok=True)
    results.to_pickle(f"{output_dir}/responses.pkl")