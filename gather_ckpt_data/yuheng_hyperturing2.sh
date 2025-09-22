# 14m, 70m, 160m

for repo_id in EleutherAI/pythia-160m EleutherAI/pythia-70m EleutherAI/pythia-14m
do
    python pretrain_helm_json2csv.py --repo_id "$repo_id"
    python pretrain_helm_csv2matrix.py --repo_id "$repo_id"
done
