import pickle
import re
import string
from huggingface_hub import snapshot_download
SEED = 42

model_nickname2helm_model_name = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "meta/llama-3-8b",
}

def create_prompts_and_answers(model_nickname, dataset, num_prompts_to_use):
    cache_dir = snapshot_download(
        repo_id="stair-lab/monkey_query_pre", 
        repo_type="dataset",
    )
    helm_model_name = model_nickname2helm_model_name[model_nickname]
    file_path = f"{cache_dir}/{helm_model_name.replace('/', '_')}_{dataset}_pre_query.pkl"
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
        
    sampled_df = df.sample(n=num_prompts_to_use, random_state=SEED).reset_index(drop=True)
    data = {
        "problems":          sampled_df["instance.input.text"].tolist(),
        "problems_indices":  sampled_df["prompt_index"].tolist(),
        "prompts":           sampled_df["request.prompt"].tolist(),
        "solutions":         sampled_df["solution"].tolist()
    }
    return data

def exact_match(response: str, solution: str) -> float:
    if not response:
        return 0
    return 1 if solution.strip() == response.strip() else 0

def normalize_text(text: str, should_remove_articles: bool = True) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    normalized_text = remove_punc(lower(text))
    if should_remove_articles:
        normalized_text = remove_articles(normalized_text)
    return white_space_fix(normalized_text)

def quasi_exact_match(solution: str, response: str) -> float:
    if not response:
        return 0
    return 1 if normalize_text(solution) == normalize_text(response) else 0
