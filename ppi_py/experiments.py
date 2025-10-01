import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import numpy as np
import pandas as pd
from ppi_py import classical_ols_pointestimate, ppi_ols_pointestimate, debias_pointestimate, imputed_ols_pointestimate

from functools import partial
from tqdm.auto import tqdm
from utils import *


def generate_data(
    N: int,
    p: int,
    noise: float = 1.0,
    sigma: float = 1.0,
    beta: float = 0.0,
    theta: np.ndarray = None,
    random_state: int = None
):
    rng = np.random.RandomState(random_state)

    if theta is None:
        theta = rng.uniform(low=-1.0, high=1.0, size=p)

    beta = np.zeros(p) if beta == 0 else rng.normal(scale=float(beta), size=p)

    X_total = rng.normal(size=(N, p))

    Y_total = X_total @ theta + noise * rng.normal(size=N)

    Y_hat = Y_total + sigma * rng.normal(size=N) + X_total @ beta

    return X_total, Y_total, Y_hat, theta


METHODS = {
    "Imputed": imputed_ols_pointestimate,
    "Classical": classical_ols_pointestimate,
    "PPI_lambda1": partial(ppi_ols_pointestimate, lam=1),
    "PPI_tuned":   partial(ppi_ols_pointestimate, lam=None),
    "Debias_Lasso": partial(debias_pointestimate, method="lasso"),
    "Debias_ElasticNet": partial(debias_pointestimate, method="elasticnet"),
    "Debias_Ridge": partial(debias_pointestimate, method="ridge"),
}

# def run(ns, ps, num_trials=30, N=5000, noise=1.0, bet=0, sig=1.0):
#     results = []
#     for p in ps:
#         _X_total, _Y_total, Yhat_total, theta = generate_data(N, p, noise=noise, beta=bet, sigma=sig)
#         for n in ns:
#             for trial in range(num_trials):
#                 idx = np.random.permutation(N)
#                 X_labeled, X_unlabeled = _X_total[idx[:n]], _X_total[idx[n:]]
#                 Y_labeled, Y_unlabeled = _Y_total[idx[:n]], _Y_total[idx[n:]]
#                 Yhat_labeled, Yhat_unlabeled = Yhat_total[idx[:n]], Yhat_total[idx[n:]]

#                 for method_name, method in METHODS.items():
#                     try:
#                         point = method(X_labeled, Y_labeled, Yhat_labeled, X_unlabeled, Yhat_unlabeled)
#                         error = (point - theta) ** 2
#                         results.append({
#                             "method": method_name, "p": p, "n": n, "trial": trial, "sigma": sig,
#                             "theta": theta.mean(),
#                             "error": error.mean(), "error_max": error.max(), "error_min": error.min()
#                         })
#                     except Exception as e:
#                         print(f"[{method_name} ERROR] p={p}, n={n}, trial={trial}: {e}")
#                         break

#     df = pd.DataFrame(results)
#     return df


def run(ns, ps, num_trials=30, N=5000, noise=1.0, bet=0, sig=1.0):
    results = []
    total_steps = len(ps) * len(ns) * num_trials * max(1, len(METHODS))
    pbar = tqdm(total=total_steps, desc="total", dynamic_ncols=True, smoothing=0.1)

    for p in ps:
        _X_total, _Y_total, Yhat_total, theta = generate_data(N, p, noise=noise, beta=bet, sigma=sig)
        for n in ns:
            for trial in range(num_trials):
                idx = np.random.permutation(N)
                X_labeled, X_unlabeled = _X_total[idx[:n]], _X_total[idx[n:]]
                Y_labeled, Y_unlabeled = _Y_total[idx[:n]], _Y_total[idx[n:]]
                Yhat_labeled, Yhat_unlabeled = Yhat_total[idx[:n]], Yhat_total[idx[n:]]

                for method_name, method in METHODS.items():
                    try:
                        point = method(X_labeled, Y_labeled, Yhat_labeled, X_unlabeled, Yhat_unlabeled)
                        point = point[0] if isinstance(point, tuple) else point
                        error = (point - theta) ** 2
                        rec = {
                            "method": method_name, "p": p, "n": n, "N": N, "trial": trial, "sigma": sig,
                            "theta": float(np.mean(theta)),
                            "error": float(np.mean(error)),
                            "error_max": float(np.max(error)),
                            "error_min": float(np.min(error)),
                        }
                        results.append(rec)
                        # 在单条进度条的后缀给出简洁状态
                        pbar.set_postfix_str(
                            f"p={p} n={n} trial={trial}",
                            refresh=False
                        )
                    except Exception as e:
                        tqdm.write(f"[{method_name} ERROR] p={p}, n={n}, trial={trial}: {e}")
                    finally:
                        pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)
