import pandas as pd
import os
import json
from datasets import load_dataset
from tqdm import tqdm

df = pd.read_parquet('data/models_gsm8k_evals_2024-10-10.parquet')
df300 = df[df['Model Nickname'].str.endswith('300B')].copy()
model_names = sorted({
    '_'.join(n.split('_')[:2])
    for n in df300['Model Nickname'].unique()
})

ds = load_dataset("openai/gsm8k", "main", split="test")

output_dir = "data/rylan_monkey"
os.makedirs(output_dir, exist_ok=True)

for model in tqdm(model_names):
    sel = df300[df300['Model Nickname'].str.startswith(model)]
    
    records = []
    for idx in sorted(sel['prompt_idx'].unique()):
        scores = sel.loc[sel['prompt_idx'] == idx, 'Score'].tolist()
        question = ds[int(idx)]['question']
        records.append({
            'is_corrects': scores,
            'question': question
        })
    
    out_path = f'GSM8K_{model}.json'
    with open(f"{output_dir}/{out_path}", 'w') as f:
        json.dump(records, f, indent=2)