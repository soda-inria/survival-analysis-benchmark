from collections import defaultdict
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
from time import perf_counter

from sklearn.model_selection import KFold

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
    subsample_train=1,
    subsample_val=1,
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

    cv_scores = defaultdict(list)

    cv = cv or KFold(shuffle=True, random_state=random_state)
    for train_idxs, val_idxs in cv.split(X):

        if subsample_train:
            n_train = len(train_idxs)
            subset_train = int(subsample_train * n_train)
            train_idxs = train_idxs[:subset_train]

        if subsample_val:
            n_val = len(val_idxs)
            subset_val = int(subsample_val * n_val)
            val_idxs = val_idxs[:subset_val]

        print(f"train set: {len(train_idxs)}, val set: {len(val_idxs)}")

        X_train, y_train, X_val, y_val = estimator.train_test_split(X, y, train_idxs, val_idxs)
        X_val, y_val = truncate_val_to_train(y_train, X_val, y_val)

        # Generate evaluation time steps as a subset of the train and val durations.
        times = get_time_grid(y_train, y_val)

        t0 = perf_counter()
        estimator.fit(X_train, y_train, times)
        t1 = perf_counter()

        scores = get_scores(estimator, y_train, X_val, y_val, times)

        # Accumulate each score into a separate list.
        for k, v in scores.items():
            cv_scores[k].append(v)
        cv_scores["training_duration"].append(t1 - t0)

        if single_fold:
            break

    print("-" * 8)
    # Compute each score mean and std
    score_keys = [
        "ibs",
        "c_index",
        #"c_index_ipcw",
        "training_duration",
        "prediction_duration",
    ]
    for k in score_keys:
        values = cv_scores.pop(k)
        k_mean, k_std = f"mean_{k}", f"std_{k}"
        cv_scores[k_mean] = np.mean(values)
        cv_scores[k_std] = np.std(values)
        print(f"{k}: {cv_scores[k_mean]:.4f} ± {cv_scores[k_std]:.4f}")
    cv_scores["n_sample_train"] = len(train_idxs)
    cv_scores["n_sample_val"] = len(val_idxs)

    save_scores(estimator.name, cv_scores)


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
    y_train_arr, y_val_arr = estimator.prepare_y_scoring(y_train, y_val)

    _, brier_scores = brier_score(y_train_arr, y_val_arr, survival_probs, times)
    ibs = integrated_brier_score(y_train_arr, y_val_arr, survival_probs, times)

    # As the C-index is expensive to compute, we only consider a subsample of our data. 
    N_sample_c_index = 50_000
    c_index = concordance_index_censored(
        y_val_arr["event"][:N_sample_c_index],
        y_val_arr["duration"][:N_sample_c_index],
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
            y_train_arr[:N_sample_c_index],
            y_val_arr[:N_sample_c_index],
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


def save_scores(name, scores):
    path = _make_path(name, create_dir=True)
    pickle.dump(scores, open(path, "wb+"))


def load_scores(name):
    path = _make_path(name, create_dir=False)
    return pickle.load(open(path, "rb"))


def _make_path(name, create_dir):
    path = Path(os.getenv("PYCOX_DATA_DIR")) / "kkbox_v1" / "results"
    if create_dir:
        path.mkdir(exist_ok=True, parents=True)
    return path / f"{name}.pkl"


def get_all_results(match_filter: str = None):
    
    results_dir = Path(os.getenv("PYCOX_DATA_DIR")) / "kkbox_v1" / "results"
    lines, tables = [], []
    match_filter = match_filter or ""
    
    for path in results_dir.iterdir():
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