# Procedure for running ising_fista.py
## Python Environment
```
conda create --name deval python=3.10 -y
conda activate deval
pip install -r requirements.txt
```

## Gather data from HELM

`json2csv.py` is the same as before except that it also reads in the `instances.json`

`csv2matrix.py` is the same as before except that it pivot the columns by `"input.text",  "references", "scenario", "benchmark"`

`irt_calibrate.py` get the z for each question and add them as a column key to the matrix, so now the 5 column keys are `"input.text",  "references", "scenario", "benchmark", "z"`

`gather_lite_data.ipynb` will get the resmat_lite_all.csv

```
cd gather_helm_data
python json2csv.py
python csv2matrix.py
python irt_calibrate.py
# run `gather_lite_data.ipynb`
cd ..
```

## fit the ising model
```
python ising_fista.py --resmat_path gather_helm_data/resmat_lite_all.csv
```



# Other Environments
## R Environment
```
conda create -n deval_R r-base=4.3 -c conda-forge -y
conda activate deval_R
conda install -c conda-forge r-psychonetrics -y
```

## Board Game Environment
```
cd boardlaw-paper-v2
conda create --name board python=3.9 -y
conda activate board
pip install -r requirements.txt
```

## Environment for automatic evaluation on model checkpoints
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
pip install hf_xet
```

## Environment for automatic evaluation on Olmo2 checkpoints
```
git clone https://github.com/sangttruong/helm
conda create -n olmo python=3.10 pip -y
conda activate olmo
pip install crfm-helm[all]
cd helm
git checkout -b auto_eval origin/auto_eval
pip uninstall crfm-helm
pip install -e .
pip install ai2-olmo
pip install vllm==0.8.1
```