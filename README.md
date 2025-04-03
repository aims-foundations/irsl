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