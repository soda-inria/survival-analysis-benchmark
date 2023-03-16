from collections import defaultdict
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
from time import perf_counter

from sklearn.model_selection import KFold, train_test_split

from sksurv.metrics import (
    brier_score,
    integrated_brier_score,
    cumulative_dynamic_auc,
    concordance_index_censored,
    concordance_index_ipcw,
)


def run_cv(
    X,
    y,
    estimator,
    subsample_train=1.0,
    subsample_val=1.0,
    cv=None,
    single_fold=False,
    random_state=42,
):
    """Run cross validation and save score.

    Parameters
    ----------
    estimator :
        Instance of an estimator to evaluate.

    kfold_tuples : tuple of tuple of ndarray,
        Data for training and evaluating.

    times : ndarray
        Timesteps used for evaluation.

    score_func : callable, default=None
        Score function that generates a dictionnary of metrics.
        If set to None, `run_cv` will call `get_score`.

    subset: dict, default=None
        The number of first rows to select for keys 'train' and 'val'.
        If `subset` is not None, one value must be set for each key.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    cv_scores = []
    cv = cv or KFold(shuffle=True, random_state=random_state)

    for train_idxs, val_idxs in cv.split(X):

        size_train = int(subsample_train * len(train_idxs))
        size_val = int(subsample_val * len(val_idxs))

        train_idxs = train_idxs[:size_train]
        val_idxs = val_idxs[:size_val]

        X_train, y_train = X[train_idxs], y[train_idxs]
        X_val, y_val = X[val_idxs], y[val_idxs]

        X_val, y_val = truncate_val_to_train(y_train, X_val, y_val)

        print(f"train set: {X_train.shape[0]}, val set: {X_val.shape[0]}")

        # Generate evaluation time steps as a subset of the train and val durations.
        times = get_time_grid(y_train, y_val)

        t0 = perf_counter()
        estimator.fit(X_train, y_train, times)
        t1 = perf_counter()

        scores = get_scores(estimator, y_train, X_val, y_val, times)
        scores["training_duration"] = t1 - t0
        cv_scores.append(scores)

        if single_fold:
            break

    print("-" * 8)
    results = {}

    # sufficient statistics
    for k in [
        "ibs", "c_index", "training_duration", "prediction_duration"
    ]:
        score_mean  = np.mean([score[k] for score in cv_scores])
        score_std = np.std([score[k] for score in cv_scores])
        results[f"mean_{k}"] = score_mean
        results[f"std_{k}"] = score_std
        print(f"{k}: {score_mean:.4f} ± {score_std:.4f}")

    # vectors
    for k in ["times", "survival_probs", "brier_scores"]:
        results[k] = [score[k] for score in cv_scores]

    results["n_sample_train"] = len(train_idxs)
    results["n_sample_val"] = len(val_idxs)

    save_scores(estimator.name, results)


def survival_to_risk_estimate(survival_probs):
    return -np.log(survival_probs + 1e-8).sum(axis=1)


def get_scores(
        estimator,
        y_train,
        X_val,
        y_val,
        times,
        use_cindex_ipcw=False,
    ):

    t0 = perf_counter()
    survival_probs = estimator.predict_survival_function(X_val, times)
    t1 = perf_counter()

    risk_estimate = survival_to_risk_estimate(survival_probs)

    _, brier_scores = brier_score(y_train, y_val, survival_probs, times)
    ibs = integrated_brier_score(y_train, y_val, survival_probs, times)

    # As the C-index is expensive to compute, we only consider a subsample of our data. 
    N_sample_c_index = 50_000
    c_index = concordance_index_censored(
        y_val["event"][:N_sample_c_index],
        y_val["duration"][:N_sample_c_index],
        risk_estimate[:N_sample_c_index],
    )[0]

    results = dict(
        brier_scores=brier_scores,
        ibs=ibs,
        times=times,
        survival_probs=survival_probs,
        c_index=c_index,
        prediction_duration=t1 - t0,
    )

    if use_cindex_ipcw:
        c_index_ipcw = concordance_index_ipcw(
            y_train[:N_sample_c_index],
            y_val[:N_sample_c_index],
            risk_estimate[:N_sample_c_index],
        )[0]
        results["c_index_ipcw"] = c_index_ipcw

    return results


def truncate_val_to_train(y_train, X_val, y_val):
    """Enforce y_val to stay below y_train upper bound"""
    out_of_bound_mask = y_train["duration"].max() <= y_val["duration"]
    return X_val[~out_of_bound_mask, :], y_val[~out_of_bound_mask]


def get_time_grid(y_train, y_val, n=100):
    y_time = np.hstack([y_train["duration"], y_val["duration"]])
    lower, upper = np.percentile(y_time, [2.5, 97.5])
    return np.linspace(lower, upper, n)


def save_scores(name, scores, create_dir=True):
    path_results = get_path_results()
    if create_dir:
        path_results.mkdir(exist_ok=True, parents=True)
    path = path_results / f"{name}.pkl"

    pickle.dump(scores, open(path, "wb+"))


def load_scores(name):
    path_results = get_path_results()
    path = path_results / f"{name}.pkl"
    return pickle.load(open(path, "rb"))


def get_path_results():
    return Path(os.getenv("PYCOX_DATA_DIR")) / "kkbox_v1" / "results"


def get_all_results(match_filter: str = None):
    """Load all results matching `match_filter`, concatenate sufficient
    statistics in `df_tables` and times, survival probs and
    brier scores vectors into `df_lines`.
    """
    path_results = get_path_results()
    lines, tables = [], []
    match_filter = match_filter or ""
    
    for path in path_results.iterdir():
        if (
            path.is_file()
            and path.suffix == ".pkl"
            and match_filter in str(path)
        ):
            result = pickle.load(open(path, "rb"))
            model_name = path.name.split(".")[0].split("_")[-1]

            line = make_row_line(result, model_name)
            table = make_row_table(result, model_name)

            lines.append(line)
            tables.append(table)

    df_tables = pd.DataFrame(tables)
    df_lines = pd.DataFrame(lines)
    
    # sort by ibs
    df_tables["ibs_tmp"] = (
        df_tables["IBS"].str.split("±")
        .str[0]
        .astype(np.float64)
    )
    df_tables = (
        df_tables.sort_values("ibs_tmp")
        .reset_index(drop=True)
        .drop(["ibs_tmp"], axis=1)
    )
    
    return df_tables, df_lines
    

def make_row_line(result, model_name):
    """Format the results of a single model into a row with
    times, survival probs and brier scores vectors for visualization
    purposes.
    """
    # times are the same across all folds, output shape: (times)
    times = result["times"][0] 
    
    # take the mean for each folds, output shape: (times)
    brier_scores = np.asarray(result["brier_scores"]).mean(axis=0)

    # arbitrarily take the first cross val to vizualize surv probs
    # output shape: (n_val, times)
    survival_probs = np.asarray(result["survival_probs"][0])
    
    return dict(
        model=model_name,
        times=times,
        brier_scores=brier_scores,
        survival_probs=survival_probs,
    )
    

def make_row_table(result, model_name):
    """Format the results of a single model into a row with
    sufficient statistics.
    """
    row = {"Method": model_name}
    
    col_displayed = {"c_index": "C_td", "ibs": "IBS"}
    for col in ["c_index", "ibs"]:
        mean_col, std_col = f"mean_{col}", f"std_{col}"
        row[col_displayed[col]] = f"{result[mean_col]:.4f} ± {result[std_col]:.4f}"
    
    for col in ["training_duration", "prediction_duration"]:
        mean_col = f"mean_{col}"
        row[col] = f"{result[mean_col]:.4f}s"
    
    for col in ["n_sample_train", "n_sample_val"]:
        row[col] = result[col]

    return row 