```
conda create --name deavl python=3.10 -y
conda activate deavl
pip install -r requirements.txt
```

```
conda create -n rpsy r-base=4.3 -c conda-forge
conda activate rpsy
conda install -c conda-forge r-psychonetrics
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