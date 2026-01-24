import os
import shutil
import gzip
import json
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from hf_olmo import OLMoTokenizerFast
from huggingface_hub import snapshot_download

ROOT_DIR = "./DataDecide-eval-instances"

local_dir = snapshot_download(
    repo_id="allenai/DataDecide-eval-instances",
    repo_type="dataset",
    allow_patterns=["requests/*"],
    cache_dir=ROOT_DIR
)

# dirs_to_visit = [ROOT_DIR]

# while dirs_to_visit:
#     current_dir = dirs_to_visit.pop()
#     with os.scandir(current_dir) as it:
#         for entry in it:
#             path = entry.path

#             if entry.is_dir():
#                 if ".cache" in path.split(os.sep):
#                     continue
#                 dirs_to_visit.append(path)
#                 continue

#             if path.endswith(".jsonl.gz"):
#                 dest = path[:-3]
#                 if os.path.exists(dest):
#                     print(f"[jsonl.gz] Skipping {dest}")
#                 else:
#                     with gzip.open(path, "rb") as src, open(dest, "wb") as dst:
#                         shutil.copyfileobj(src, dst)
#                     print(f"[jsonl.gz] Wrote {dest}")

# {
#     "request_type": "loglikelihood",
#     "doc": {
#         "id": "Mercury_7175875",
#         "query": "Question: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nAnswer:",
#         "choices": ["Planetary density will decrease.", "Planetary years will become longer.", "Planetary days will become shorter.", "Planetary gravity will become stronger."],
#         "gold": 2
#     }, 
#     "request": {
#         "context": "Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\nAnswer: dry palms\n\nQuestion: Which of the following statements best explains why magnets usually stick to a refrigerator door?\nAnswer: The refrigerator door contains iron.\n\nQuestion: A fold observed in layers of sedimentary rock most likely resulted from the\nAnswer: converging of crustal plates.\n\nQuestion: Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the Mesozoic era?\nAnswer: impact of an asteroid created dust that blocked the sunlight\n\nQuestion: Which of the following is a trait that a dog does NOT inherit from its parents?\nAnswer: the size of its appetite\n\nQuestion: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nAnswer:",
#         "continuation": " Planetary density will decrease."
#     },
#     "idx": 0, "task_name": "arc_challenge", "doc_id": 0, "native_id": "Mercury_7175875", "label": 2
# }
# {
#     "request_type": "loglikelihood",
#     "doc": {
#         "id": "Mercury_7175875",
#         "query": "Question: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nAnswer:",
#         "choices": ["Planetary density will decrease.", "Planetary years will become longer.", "Planetary days will become shorter.", "Planetary gravity will become stronger."],
#         "gold": 2
#     }, 
#     "request": {
#         "context": "Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\nAnswer: dry palms\n\nQuestion: Which of the following statements best explains why magnets usually stick to a refrigerator door?\nAnswer: The refrigerator door contains iron.\n\nQuestion: A fold observed in layers of sedimentary rock most likely resulted from the\nAnswer: converging of crustal plates.\n\nQuestion: Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the Mesozoic era?\nAnswer: impact of an asteroid created dust that blocked the sunlight\n\nQuestion: Which of the following is a trait that a dog does NOT inherit from its parents?\nAnswer: the size of its appetite\n\nQuestion: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nAnswer:",
#         "continuation": " Planetary years will become longer."
#     }, 
#     "idx": 1, "task_name": "arc_challenge", "doc_id": 0, "native_id": "Mercury_7175875", "label": 2
# }
# {"request_type": "loglikelihood", "doc": {"id": "Mercury_7175875", ...}}
# {"request_type": "loglikelihood", "doc": {"id": "Mercury_7175875", ...}}

# {
#     "request_type": "loglikelihood",
#     "doc": {
#         "id": "Mercury_SC_409171",
#         "query": "Question: A group of engineers wanted to know how different building designs would respond during an earthquake. They made several models of buildings and tested each for its ability to withstand earthquake conditions. Which will most likely result from testing different building designs?\nAnswer:",
#         "choices": ["buildings will be built faster", "buildings will be made safer", "building designs will look nicer", "building materials will be cheaper"], 
#         "gold": 1
#     }, 
#     "request": {
#         "context": "Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\nAnswer: dry palms\n\nQuestion: Which of the following statements best explains why magnets usually stick to a refrigerator door?\nAnswer: The refrigerator door contains iron.\n\nQuestion: A fold observed in layers of sedimentary rock most likely resulted from the\nAnswer: converging of crustal plates.\n\nQuestion: Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the Mesozoic era?\nAnswer: impact of an asteroid created dust that blocked the sunlight\n\nQuestion: Which of the following is a trait that a dog does NOT inherit from its parents?\nAnswer: the size of its appetite\n\nQuestion: A group of engineers wanted to know how different building designs would respond during an earthquake. They made several models of buildings and tested each for its ability to withstand earthquake conditions. Which will most likely result from testing different building designs?\nAnswer:", "continuation": " buildings will be built faster"
#     }, 
#     "idx": 0, "task_name": "arc_challenge", "doc_id": 1, "native_id": "Mercury_SC_409171", "label": 1
# }

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))
from utils import MODEL2PARA

tokenizer = OLMoTokenizerFast.from_pretrained("allenai/DataDecide-dclm-baseline-4M")
requests_root = os.path.join(local_dir, "requests")
eval_tokens = 0
files = []
for dirpath, _, filenames in os.walk(requests_root):
    for name in filenames:
        if name.endswith(".jsonl"):
            files.append(os.path.join(dirpath, name))
for path in tqdm(files, desc="requests"):
    f = open(path, "r", encoding="utf-8")
    with f:
        for line in f:
            row = json.loads(line)
            doc = row.get("doc", {})
            query = doc.get("query", "")
            choices = doc.get("choices", [])
            text = query + " " + " ".join(choices)
            eval_tokens += len(tokenizer.encode(text, add_special_tokens=False))

print(f"eval_tokens={eval_tokens}")

long_path = BASE_DIR / "data" / "6_long.parquet"
long_df = pd.read_parquet(long_path).reset_index()
para = long_df["model_size"].map(MODEL2PARA)
eval_flop = (2 * para * eval_tokens).sum()
pretrain_flop = long_df["FLOP"].sum(skipna=True)
proportion = eval_flop / pretrain_flop
print(f"eval_flop={eval_flop:.3e}")
print(f"pretrain_flop={pretrain_flop:.3e}")
print(f"proportion={proportion:.3e}")

pd.set_option("display.max_rows", None)
flop_non_na = long_df["FLOP"].notna()
print(
    long_df.loc[flop_non_na, ["model_data_mix", "model_size", "FLOP"]]
    .drop_duplicates(subset=["model_size"])
)
