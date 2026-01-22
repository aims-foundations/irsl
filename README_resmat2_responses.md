# IRSL Test-Time ResMAT2 Full Responses

This dataset contains full text responses with reasoning chains from large language models evaluated on various benchmarks at temperature 1.0. This is the complete response data for **ResMAT2** used in the paper "Interpretable, Reliable, and Scalable Laws for LLMs" (ICML 2026).

## Dataset Overview

- **Models**: 12 state-of-the-art LLMs (2024-2025)
- **Benchmarks**: 9 diverse tasks (BBQ, GSM8k, Commonsense, LegalBench, MedQA, LSAT, Legal Support, MMLU, MATH)
- **Questions**: ~120 total questions (varies by benchmark)
- **Responses per question**: 2,560 independent samples
- **Temperature**: 0.6 (allows for diverse reasoning)
- **Max tokens**: 512
- **Prompting**: Zero-shot with chain-of-thought instructions

## Why This Dataset?

The original ResMAT2 tensor (`stair-lab/irsl_testtime_resmat2`) only contains binary correctness scores (0/1). This dataset preserves the **full text responses** that were expensive to collect, enabling:

- 🔍 Analysis of chain-of-thought reasoning patterns
- 🤖 Comparison of reasoning strategies across models
- 📊 Study of response diversity at non-zero temperature
- 🐛 Debugging error patterns in model behavior
- 🔬 Research on LLM reasoning capabilities

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
| `prompt` | str | Complete prompt with instructions |
| `responses` | List[str] | 2,560 full model responses |
| `scores` | List[int] | 2,560 binary correctness scores (0/1) |

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
    repo_id="stair-lab/irsl_testtime_resmat2_responses",
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

# Show first response
print(f"\nFirst response:\n{question_data['responses'][0]}")
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
| Qwen2.5-72B-Instruct | 72B | Latest Qwen |
| Qwen2.5-Math-72B-Instruct | 72B | Math-specialized |
| gpt-4o-mini-2024-07-18 | - | OpenAI GPT-4o mini |
| claude-3-5-sonnet-20241022 | - | Anthropic Claude 3.5 |

## Available Benchmarks

| Benchmark | Questions | Task Type | Evaluation |
|-----------|-----------|-----------|------------|
| BBQ | ~96 | Bias detection | Quasi-exact match |
| GSM8k | ~96 | Math word problems | Number extraction |
| Commonsense | ~96 | Multiple-choice QA | Exact match |
| LegalBench | ~96 | Legal classification | Quasi-exact match |
| MedQA | ~96 | Medical QA | Quasi-exact match |
| LSAT_QA | ~96 | Logical reasoning | Quasi-exact match |
| Legal_Support | ~96 | Legal reasoning | Quasi-exact match |
| MMLU | ~96 | Multi-task QA | Exact match |
| MATH | ~96 | Advanced math | Chain-of-thought match |

## Example Analyses

### Compare Reasoning Patterns

```python
# Load responses from two models
df_deepseek = pd.read_parquet("DeepSeek-R1-Distill-Llama-8B_bbq.parquet")
df_llama = pd.read_parquet("Meta-Llama-3-70B-Instruct_bbq.parquet")

# Compare on same question
q_idx = 0
print(f"Question: {df_deepseek.iloc[q_idx]['question']}\n")

print("DeepSeek-R1 response:")
print(df_deepseek.iloc[q_idx]['responses'][0])
print(f"Score: {df_deepseek.iloc[q_idx]['scores'][0]}\n")

print("Llama-3-70B response:")
print(df_llama.iloc[q_idx]['responses'][0])
print(f"Score: {df_llama.iloc[q_idx]['scores'][0]}")
```

### Analyze Response Diversity

```python
import numpy as np

# Compute response variance for each question
for idx, row in df.iterrows():
    scores = row['scores']
    variance = np.var(scores)
    accuracy = np.mean(scores)

    print(f"Q{idx}: Acc={accuracy:.1%}, Var={variance:.4f}")
```

### Find Error Patterns

```python
# Find questions where model frequently gets wrong
for idx, row in df.iterrows():
    accuracy = sum(row['scores']) / len(row['scores'])

    if accuracy < 0.5:  # Less than 50% accuracy
        print(f"\nDifficult Question (Acc={accuracy:.1%}):")
        print(row['question'])

        # Show incorrect responses
        incorrect_responses = [
            r for r, s in zip(row['responses'], row['scores']) if s == 0
        ]
        print(f"\nExample incorrect response:")
        print(incorrect_responses[0][:500])
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
df = load_model_responses("resmat2", "DeepSeek-R1-Distill-Llama-8B", "bbq")

# Show sample responses
show_sample_responses(df, question_idx=0, n=5)

# Compute statistics
stats = compute_response_statistics(df)
print(stats.sort_values('accuracy'))
```

## Dataset Statistics

- **Total size**: ~1.5-2 GB compressed
- **Total responses**: ~350,000 (12 models × ~100 questions × 2,560 samples)
- **Average response length**: ~150-200 tokens
- **Compression**: Snappy (fast decompression)

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

- **Binary scores only**: `stair-lab/irsl_testtime_resmat2` (tensor format, 12 models × 120 questions × 2,560 samples)
- **ResMAT1 full responses**: `stair-lab/irsl_testtime_resmat1_responses` (8 models × 552 questions × 10,000 samples)
- **Source parquet batches**: `stair-lab/denoise_eval_query` (fragmented batch files)

## License

This dataset is released under the same license as the original ICML 2026 paper codebase.

## Contact

For questions or issues, please open an issue on the GitHub repository:
https://github.com/stair-lab/generalized-scaling-laws
