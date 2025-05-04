import numpy as np
np.random.seed(42)
import pickle
import os
import pandas as pd
import json
from huggingface_hub import HfApi
from huggingface_hub import login

def get_solution(row):
    # Parse the references to find the correct answer text
    refs = json.loads(row["instance.references"])
    correct_text = None
    for ref in refs:
        if "correct" in ref.get("tags", []):
            correct_text = ref["output"]["text"]
            break
    if correct_text is None:
        raise ValueError(f"No correct solution found for row with index {row.name}")

    # Dynamically identify all output_mapping columns (e.g., output_mapping.A, output_mapping.B, etc.)
    choice_columns = [col for col in row.index if col.startswith("output_mapping.")]

    # Iterate through the available choices to find the matching letter
    for col in choice_columns:
        if row[col] == correct_text:
            return col.split(".")[-1]  # Extract the letter (e.g., 'A' from 'output_mapping.A')

    raise ValueError(f"Correct answer text not found among choices for row with index {row.name}")


if __name__ == "__main__":
    model_name = "meta/llama-3-8b"
    
    # benchmark_name = "mmlu"
    # scenario_name = "mmlu"
    
    # benchmark_name = "lite"
    # scenario_name = "commonsense"
    # scenario_name = "med_qa"
    # scenario_name = "legalbench"
    
    benchmark_name = "classic"
    # scenario_name = "legal_support"
    scenario_name = "bbq"
    # scenario_name = "lsat_qa"
    
    with open(f"data/gather_helm_data/responses_monkey_{benchmark_name}.pkl", "rb") as f:
        results_full = pickle.load(f)
    
    results_full = results_full.sample(frac=1).reset_index(drop=True)
    all_choice_columns = [col for col in results_full.columns if col.startswith("output_mapping.")]
    scenario_mask = results_full["scenario"] == scenario_name
    choice_columns = [
        col for col in all_choice_columns
        if not results_full.loc[scenario_mask, col].isna().all()
    ]
    results = results_full[
        ["request.model", "instance.input.text", "request.prompt", "instance.references", "scenario", "benchmark", "dicho_score"] + choice_columns
    ]
    results = results.dropna(subset=["request.model", "instance.input.text", "request.prompt", "instance.references", "scenario", "benchmark", "dicho_score"])
    
    # drop the dicho_score of 0.5
    results = results[results["dicho_score"] != 0.5]
    results["dicho_score"] = results["dicho_score"].astype(bool)
    assert results["dicho_score"].isin([0, 1]).all()
    
    # drop duplicate rows
    results = results.drop_duplicates(subset=["request.model", "instance.input.text", "scenario", "benchmark"], keep='first')
    print(f"non-duplicate percentage:{results.shape[0]/results_full.shape[0]}")
    
    # Count the number of unique instance.input.text for each request.model
    model_prompt_counts = results.groupby('request.model', observed=True)['instance.input.text'].nunique()
    # Count the number of unique request.model for each instance.input.text
    prompt_model_counts = results.groupby('instance.input.text', observed=True)['request.model'].nunique()
    # Identify models with at least 30 unique prompts and prompts with at least 30 unique models
    models_to_keep = model_prompt_counts[model_prompt_counts >= 30].index
    prompts_to_keep = prompt_model_counts[prompt_model_counts >= 30].index
    # Filter the DataFrame accordingly
    results = results[
        results['request.model'].isin(models_to_keep) &
        results['instance.input.text'].isin(prompts_to_keep)
    ]
    breakpoint() # has 01

    # filter out one scenario, delete all 0 or all 1 cols
    results = results[
        (results["benchmark"] == benchmark_name) &
        (results["scenario"] == scenario_name)
    ]
    breakpoint() # all 1??
    
    results["dicho_score"] = results["dicho_score"].astype(int)
    results = results.groupby(["instance.input.text", "scenario", "benchmark"], observed=False).filter(lambda grp: grp["dicho_score"].nunique() > 1)
    
    # filter model row, make sure number of few shot of the request fit in the model's max input token length
    if benchmark_name=="classic":
        # classic does not have llama-3-8b, this model also has 8192 context length, same as llama-3-8b
        context_model_name = "anthropic/stanford-online-all-v4-s3"
    else:
        context_model_name = model_name
    results = results[(results["request.model"] == context_model_name)]
    print(len(results))
    
    # get "solution column", should be ABCD
    results["solution"] = results.apply(get_solution, axis=1)
    
    # save result
    pre_query = results.copy()
    pre_query = pre_query[["instance.input.text", "request.prompt", "solution", "scenario", "benchmark"]]
    pre_query = pre_query.sort_values('instance.input.text').reset_index(drop=True)
    pre_query['prompt_index'] = range(len(pre_query))
    print(pre_query.head())
    out_dir = 'data/monkey_query/pre_query'
    os.makedirs(out_dir, exist_ok=True)
    safe_model_name = model_name.replace('/', '_')
    out_path = f"{out_dir}/{safe_model_name}_{scenario_name}_pre_query.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(pre_query, f)

    login()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=out_path,
        path_in_repo=out_path.split('/')[-1],
        repo_id="stair-lab/monkey_query_pre",
        repo_type="dataset",
    )
        

    