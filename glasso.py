import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils import resample
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()

def binary_neighborhood_selection(X, max_workers=4):
    """
    Stability selection with bootstrapped sparse logistic regression (LogisticRegressionCV).

    Args:
        X: (N, P) binary matrix (entries should be 0 or 1).

    Returns:
        edge_freq: (P, P) matrix of edge selection frequencies.
    """
    N, P = X.shape
    n_bootstraps = 100

    def process_bootstrap(b):
        X_b = resample(X, replace=True, n_samples=N)
        local_edge_counts = np.zeros((P, P))

        def process_feature(j):
            y = X_b[:, j]
            X_others = np.delete(X_b, j, axis=1)
            feature_indices = [i for i in range(P) if i != j]

            # Logistic regression with L1 regularization and cross-validation
            model = LogisticRegressionCV(
                penalty='l1',
                solver='saga',
                cv=10,
                scoring='neg_log_loss',
                max_iter=10000,
                fit_intercept=False,
                Cs=10,  # number of inverse regularization values
                tol=1e-4,
            )

            try:
                model.fit(X_others, y)
                coefs = model.coef_[0] # shape (P - 1,)
            except Exception as e:
                # Catch numerical issues (e.g. perfect separation) and skip
                coefs = np.zeros(P - 1)

            # Collect the feature indices for which the coefficient is nonzero.
            selected_edges = []
            for idx, coef in zip(feature_indices, coefs):
                if coef != 0:
                    selected_edges.append((j, idx))
            return selected_edges

        with ThreadPoolExecutor() as executor_inner:
            results = list(executor_inner.map(process_feature, range(P)))

        for selected_edges in results:
            for (j, idx) in selected_edges:
                local_edge_counts[j, idx] += 1

        return local_edge_counts

    total_edge_counts = np.zeros((P, P))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_bootstrap, b) for b in range(n_bootstraps)]
        for future in tqdm(as_completed(futures), total=n_bootstraps, desc="Bootstraps"):
            local_edge_counts = future.result()
            total_edge_counts += local_edge_counts

    edge_freq = total_edge_counts / n_bootstraps
    edge_freq = np.maximum(edge_freq, edge_freq.T)

    return edge_freq

if __name__ == "__main__":
    file_name = 'gsm_hard_easy_200.csv'
    X = pd.read_csv(file_name).values
    edge_freq = binary_neighborhood_selection(X)
    with open('edge_freq.pkl', 'wb') as f:
        pickle.dump(edge_freq, f)
    
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.subplot(1, 2, 1)
        plt.imshow(edge_freq, cmap='Blues')
        plt.title("Edge Frequencies")

        plt.subplot(1, 2, 2)
        adj_est = (edge_freq >= 0.90).astype(int)
        plt.imshow(adj_est, cmap='Greys')
        plt.title("Recovered Graph")

        plt.tight_layout()
        plt.savefig(f"result/glasso_{file_name}.png", dpi=300, bbox_inches="tight")