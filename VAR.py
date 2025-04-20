import numpy as np
import pandas as pd
from sktime.forecasting.var import VAR as SKVAR
from sktime.forecasting.base import ForecastingHorizon
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# reproducibility
np.random.seed(42)

# Load data (no headers)
Y = pd.read_csv("gather_ckpt_data/legalbench_subset_pythia-6.9b_50.csv", header=None)

n_obs   = len(Y)
n_test  = int(0.2 * n_obs)
maxlags = 1

# only allow test times t >= maxlags+1
all_indices  = np.arange(maxlags+1, n_obs)
test_indices = np.sort(np.random.choice(all_indices, size=n_test, replace=False))
print(test_indices)

aucs = []

for t in tqdm(test_indices):
    # train on all data up to t-1 (so you have at least maxlags+1 rows)
    y_train = Y.iloc[:t]

    # fit 1‑lag VAR
    model = SKVAR(maxlags=maxlags, trend="c", verbose=False)
    model.fit(y_train)

    # one‑step ahead forecast
    fh     = ForecastingHorizon([1], is_relative=True)
    y_pred = model.predict(fh).values.ravel()

    # true at time t
    y_true = Y.iloc[t].values.ravel()

    aucs.append(roc_auc_score(y_true, y_pred))

print(" ".join(f"{auc:.2f}" for auc in aucs))

avg_auc = np.nanmean(aucs)
print(f"Average out‑of‑sample ROC AUC over {len(aucs)} random test points: {avg_auc:.2f}")
