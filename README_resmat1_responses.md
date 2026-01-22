# IRSL Test-Time ResMAT1 Full Responses

This dataset contains full text responses with reasoning chains from large language models evaluated on various benchmarks at temperature 1.0. This is the complete response data for **ResMAT1** used in the paper "Interpretable, Reliable, and Scalable Laws for LLMs" (ICML 2026).

## Dataset Overview

- **Models**: 8 LLMs (2024)
- **Benchmarks**: 6 diverse tasks (Commonsense, LegalBench, MedQA, BBQ, LSAT, Legal Support)
- **Questions**: 552 total questions
- **Responses per question**: 10,000 independent samples
- **Temperature**: 1.0 (maximum diversity)
- **Max tokens**: 512
- **Prompting**: Few-shot with examples

## Why This Dataset?

The original ResMAT1 tensor (`stair-lab/irsl_testtime_resmat1`) only contains binary correctness scores (0/1). This dataset preserves the **full text responses** that were expensive to collect, enabling:

- 🔍 Deep analysis of test-time scaling behavior
- 🤖 Study of response diversity with 10,000 samples
- 📊 Statistical analysis of reasoning patterns
- 🐛 Comprehensive error pattern analysis
- 🔬 Research on temperature effects at T=1.0

## Dataset Structure

Each parquet file represents one (model, benchmark) combination:

```
{model_name}_{benchmark_name}.parquet
```

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `question` | str | Full question text |
| `question_idx` | int | Original question index |
| `prompt` | str | Complete few-shot prompt |
| `responses` | List[str] | 10,000 full model responses |
| `scores` | List[int] | 10,000 binary correctness scores (0/1) |

## Quick Start

### Installation

```bash
pip install pandas pyarrow huggingface-hub
```

### Loading Data

```python
from huggingface_hub import hf_hub_download
import pandas as pd

# Download and load responses for DeepSeek-R1 on BBQ
file_path = hf_hub_download(
    repo_id="stair-lab/irsl_testtime_resmat1_responses",
    filename="DeepSeek-R1-Distill-Llama-8B_bbq.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(file_path)

print(f"Loaded {len(df)} questions")
print(f"Each question has {len(df.iloc[0]['responses'])} responses")
```

### Accessing Responses

```python
# Get responses for first question
question_data = df.iloc[0]

print(f"Question: {question_data['question']}")
print(f"Accuracy: {sum(question_data['scores']) / len(question_data['scores']):.1%}")

# Show first few responses
for i in range(5):
    print(f"\nResponse {i+1} (Score: {question_data['scores'][i]}):")
    print(question_data['responses'][i][:300])  # First 300 chars
```

## Available Models

| Model | Parameters | Notes |
|-------|------------|-------|
| DeepSeek-R1-Distill-Llama-8B | 8B | Chain-of-thought reasoning |
| DeepSeek-V2-Lite-Chat | ~16B | Conversational model |
| Meta-Llama-3-8B-Instruct | 8B | Instruction-tuned |
| Meta-Llama-3-70B-Instruct | 70B | Large instruction-tuned |
| Qwen3-8B | 8B | Qwen3 series |
| Qwen3-14B | 14B | |
| Qwen3-32B | 32B | |
| gemma-3-27b-it | 27B | Google Gemma |

## Available Benchmarks

| Benchmark | Questions | Task Type | Evaluation |
|-----------|-----------|-----------|------------|
| Commonsense | 92 | Multiple-choice QA | Exact match |
| LegalBench | 92 | Legal classification | Quasi-exact match |
| MedQA | 92 | Medical QA | Quasi-exact match |
| BBQ | 92 | Bias detection | Quasi-exact match |
| LSAT_QA | 92 | Logical reasoning | Quasi-exact match |
| Legal_Support | 92 | Legal reasoning | Quasi-exact match |

Each benchmark is grouped into:
- **Lite benchmarks**: Commonsense, LegalBench, MedQA (factual tasks)
- **Classic benchmarks**: BBQ, LSAT_QA, Legal_Support (reasoning tasks)

## Example Analyses

### Test-Time Scaling Analysis

```python
import numpy as np

# Analyze how accuracy changes with number of samples
question_data = df.iloc[0]
scores = question_data['scores']

# Compute pass@k for different k values
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]

for k in sample_sizes:
    accuracy_at_k = np.mean(scores[:k])
    stderr_at_k = np.std(scores[:k]) / np.sqrt(k)

    print(f"Pass@{k:5d}: {accuracy_at_k:.3f} ± {stderr_at_k:.3f}")
```

### Response Diversity at T=1.0

```python
# Count unique responses
for idx, row in df.iterrows():
    unique_responses = len(set(row['responses']))
    total_responses = len(row['responses'])
    diversity_ratio = unique_responses / total_responses

    print(f"Q{idx}: {unique_responses}/{total_responses} unique ({diversity_ratio:.1%})")
```

### Statistical Significance of Difficulty

```python
import scipy.stats as stats

# Compare two models statistically on same question
q_idx = 0

df_model1 = pd.read_parquet("DeepSeek-R1-Distill-Llama-8B_bbq.parquet")
df_model2 = pd.read_parquet("Meta-Llama-3-70B-Instruct_bbq.parquet")

scores1 = df_model1.iloc[q_idx]['scores']
scores2 = df_model2.iloc[q_idx]['scores']

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(scores1, scores2)

print(f"Model 1 accuracy: {np.mean(scores1):.1%}")
print(f"Model 2 accuracy: {np.mean(scores2):.1%}")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.6f}")
print(f"Significant: {p_value < 0.05}")
```

### Analyze Error Correlations

```python
# Are the same responses incorrect across models?
models = [
    "DeepSeek-R1-Distill-Llama-8B",
    "Meta-Llama-3-70B-Instruct"
]

dfs = {m: pd.read_parquet(f"{m}_bbq.parquet") for m in models}

# For each question, compute error correlation
for q_idx in range(len(dfs[models[0]])):
    scores = {m: np.array(dfs[m].iloc[q_idx]['scores']) for m in models}

    # Compute correlation of errors
    correlation = np.corrcoef(scores[models[0]], scores[models[1]])[0, 1]

    print(f"Q{q_idx}: Error correlation = {correlation:.3f}")
```

## Helper Functions

For easier data access, use the provided utility functions:

```python
# Download response_utils.py from the main repository
from response_utils import (
    load_model_responses,
    get_question_responses,
    show_sample_responses,
    compute_response_statistics,
    compare_model_responses,
)

# Load data
df = load_model_responses("resmat1", "DeepSeek-R1-Distill-Llama-8B", "bbq")

# Show sample responses
show_sample_responses(df, question_idx=0, n=10)

# Compute statistics
stats = compute_response_statistics(df)
print(stats.sort_values('accuracy'))

# Compare models
compare_model_responses(
    "resmat1",
    ["DeepSeek-R1-Distill-Llama-8B", "Meta-Llama-3-70B-Instruct"],
    "bbq",
    question_idx=0,
    n_samples=5
)
```

## Dataset Statistics

- **Total size**: ~12-18 GB compressed
- **Total responses**: ~44 million (8 models × 552 questions × 10,000 samples)
- **Average response length**: ~150-200 tokens
- **Compression**: Snappy (fast decompression)
- **Temperature**: 1.0 (maximum diversity)

## Comparison with ResMAT2

| Feature | ResMAT1 | ResMAT2 |
|---------|---------|---------|
| Models | 8 | 12 |
| Questions | 552 | ~120 |
| Samples/question | 10,000 | 2,560 |
| Temperature | 1.0 | 0.6 |
| Prompting | Few-shot | Zero-shot + CoT |
| Total responses | 44M | 350K |
| Size | ~15 GB | ~2 GB |
| Purpose | Test-time scaling validation | Recent model comparison |

## Citation

If you use this dataset, please cite our paper:

```bibtex
@inproceedings{anonymous2026interpretable,
  title={Interpretable, Reliable, and Scalable Laws for LLMs},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Related Datasets

- **Binary scores only**: `stair-lab/irsl_testtime_resmat1` (tensor format, 8 models × 552 questions × 10,000 samples)
- **ResMAT2 full responses**: `stair-lab/irsl_testtime_resmat2_responses` (12 models × ~120 questions × 2,560 samples)
- **HELM z-scores**: Embedded in `irsl_testtime_resmat1` tensor

## License

This dataset is released under the same license as the original ICML 2026 paper codebase.

## Contact

For questions or issues, please open an issue on the GitHub repository:
https://github.com/stair-lab/generalized-scaling-laws
