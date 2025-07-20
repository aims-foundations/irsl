from datasets import load_dataset
from vllm import LLM, SamplingParams

engine_kwargs = {
    "model": "Qwen/Qwen3-8B",
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.95,
    "enable_prefix_caching": True,
    "trust_remote_code": True
}

model = LLM(**engine_kwargs)

model_sampling_params = SamplingParams(
    n=10,
    temperature=0.6,
    max_tokens=256,
    seed=42
)

base_prompt = """You are an expert in math. When given a question:
- **Show your work.** Write calculations concisely.
- **End with “The answer is X”.** Replace X with the final numeric answer.
- **Do not include any currency symbols (e.g. “$”) or time suffixes (e.g. “:00 AM”/“:00 PM”) in X.** X must be purely numeric.
- **Do not include commas in X.** Use “70000” instead of “70,000”.
- **Do not output X with decimal places.** Do not use formats like 18.0 or 18.00; use “18” instead.
- **Do not add any content after “The answer is X”.**

Now solve this problem: """

dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
questions = dataset.select(range(3))["question"]
answers = dataset.select(range(3))["answer"]

for idx, (question, answer) in enumerate(zip(questions, answers), start=1):
    prompt = f"{base_prompt}{question}"
    requests_outputs = model.generate(prompts=[prompt], sampling_params=model_sampling_params)
    responses = [
        output.text
        for request_output in requests_outputs
        for output in request_output.outputs
    ]
    print(f"\n=== Question {idx} ===\n{question}\n")
    print(f"\n=== Answer {idx} ===\n{answer}\n")
    for i, response in enumerate(responses, start=1):
        print(f"Response {i}:\n{response.strip()}\n")
