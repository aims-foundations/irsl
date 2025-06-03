import pandas as pd
import os
import json
from datasets import load_dataset
from tqdm import tqdm
import glob
from huggingface_hub import HfApi

# dataset = "GSM8K"
# if dataset == "MATH":
#     configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
#     ds_list = [
#         load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
#         for cfg in configs
#     ]
#     ds = concatenate_datasets(ds_list)
#     df = pd.read_parquet('data/models_math_evals_2024-10-10.parquet')
#     keyname = 'problem'

# elif dataset == "GSM8K":
ds = load_dataset("openai/gsm8k", "main", split="test")
df = pd.read_parquet('../../data/models_gsm8k_evals_2024-10-10.parquet')
keyname = 'question'

df300 = df[df['Model Nickname'].str.endswith('300B')].copy()
model_names = sorted({
    '_'.join(n.split('_')[:2])
    for n in df300['Model Nickname'].unique()
})

output_dir = "../../data/rylan_monkey"
os.makedirs(output_dir, exist_ok=True)

for model in tqdm(model_names):
    sel = df300[df300['Model Nickname'].str.startswith(model)]
    
    records = []
    for idx in sorted(sel['prompt_idx'].unique()):
        scores = sel.loc[sel['prompt_idx'] == idx, 'Score'].tolist()
        # if len(scores) == 128 and set(scores) == {0}:
        #     scores = [0] * 10000
        # elif len(scores) == 128 and set(scores) != {0}:
            # raise ValueError(f"Unexpected case at model={model}, prompt_idx={idx}: len={len(scores)}, unique_scores={set(scores)}")

        question = ds[int(idx)][keyname]
        records.append({
            'is_corrects': scores,
            'question': question,
            # 'prompt_idx': int(idx)
        })
    
    out_path = f'{model}_gsm.json'
    with open(f"{output_dir}/{out_path}", 'w') as f:
        json.dump(records, f, indent=2)
        
    api = HfApi()
    rylan_query_list = glob.glob("../../data/rylan_monkey/*gsm.json")
    for file_path in rylan_query_list:
        filename = os.path.basename(file_path)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{filename}",
            repo_id="stair-lab/monkey_queries",
            repo_type="dataset",
        )
