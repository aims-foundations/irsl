# Python Environment
```
conda create --name deval python=3.10 -y
conda activate deval
pip install -r requirements.txt
```

# Pretrain binary IRT
Figure 1: `pretrain_cat_helm_binary.py`
Total Variation: `pretrain_metric_tv_binary.py`
Figure 4,7: `downstream_scaleup.py` -> `downstream_scaleup_plot.py` -> `downstream_scaleup_plot_heatmap.py`

# Pretrain beta IRT
Figure 2: `pretrain_organize_and_plotnaivelaw_resmat2.py`
Figure 3: `pretrain_cat_resmat2.py` -> `pretrain_metric_resmat2.py`

# Test-time beta IRT
Figure 5,8,9,10: `testtime_calibrate.py` -> `testtime_cat.py` -> `testtime_aggregate_results.py`
Figure 6: `monkey_analysis_denoise_data.py`