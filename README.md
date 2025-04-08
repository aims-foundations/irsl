Downloading GSM8K Responses from HELM Lite.
Follow [Google's installation](https://cloud.google.com/sdk/docs/install) instructions to install gcloud.
Then run:
```
export LOCAL_BENCHMARK_OUTPUT_PATH=../data/gather_helm_data/lite
mkdir -p $LOCAL_BENCHMARK_OUTPUT_PATH
export GCS_BENCHMARK_OUTPUT_PATH=gs://crfm-helm-public/lite/benchmark_output
gcloud storage rsync -r $GCS_BENCHMARK_OUTPUT_PATH $LOCAL_BENCHMARK_OUTPUT_PATH
```
Ref: [CRFM HELM - Downloading Raw Results](https://crfm-helm.readthedocs.io/en/latest/downloading_raw_results/)

```
conda create --name deval python=3.10 -y
conda activate deval
pip install -r requirements.txt
```

```
conda create -n deval_R r-base=4.3 -c conda-forge -y
conda activate deval_R
conda install -c conda-forge r-psychonetrics -y
```

```
git clone https://github.com/sangttruong/helm
conda create -n crfm-helm python=3.10 pip -y
conda activate crfm-helm
pip install crfm-helm[all]
cd helm
git checkout -b auto_eval origin/auto_eval
pip uninstall crfm-helm
pip install -e .
pip install vllm==0.7.3
```

For Olmo2
```
git clone https://github.com/sangttruong/helm
conda create -n olmo python=3.10 pip -y
conda activate olmo
pip install crfm-helm[all]
cd helm
git checkout -b auto_eval origin/auto_eval
pip uninstall crfm-helm
pip install -e .
pip install vllm==0.8.1
```