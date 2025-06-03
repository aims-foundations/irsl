import pandas as pd
import os
import json
from datasets import load_dataset
from tqdm import tqdm
from datasets import concatenate_datasets

dataset = "GSM8K"
if dataset == "MATH":
    configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    ds_list = [
        load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
        for cfg in configs
    ]
    ds = concatenate_datasets(ds_list)
    df = pd.read_parquet('data/models_math_evals_2024-10-10.parquet')
    keyname = 'problem'

elif dataset == "GSM8K":
    ds = load_dataset("openai/gsm8k", "main", split="test")
    # df = pd.read_parquet('../../data/models_gsm8k_evals_2024-10-10.parquet')
    keyname = 'question'

df300 = df[df['Model Nickname'].str.endswith('300B')].copy()
model_names = sorted({
    '_'.join(n.split('_')[:2])
    for n in df300['Model Nickname'].unique()
})

output_dir = "data/rylan_monkey"
os.makedirs(output_dir, exist_ok=True)

for model in tqdm(model_names):
    sel = df300[df300['Model Nickname'].str.startswith(model)]
    
    records = []
    diff_lens = []
    for idx in sorted(sel['prompt_idx'].unique()):
        scores = sel.loc[sel['prompt_idx'] == idx, 'Score'].tolist()[:10000]
        diff_lens.append(len(sel.loc[sel['prompt_idx'] == idx, 'Score'].tolist()))
        # if len(scores) == 128 and set(scores) == {0}:
        #     scores = [0] * 10000
        # elif len(scores) == 128 and set(scores) != {0}:
            # raise ValueError(f"Unexpected case at model={model}, prompt_idx={idx}: len={len(scores)}, unique_scores={set(scores)}")

        question = ds[int(idx)][keyname]
        records.append({
            'is_corrects': scores,
            'question': question,
            'prompt_idx': int(idx)
        })
    print(model, set(diff_lens))
    
    out_path = f'{dataset}_{model}.json'
    with open(f"{output_dir}/{out_path}", 'w') as f:
        json.dump(records, f, indent=2)