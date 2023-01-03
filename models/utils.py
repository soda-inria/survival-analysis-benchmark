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
    model_name,
    score_func=None,
    subset=None,
    fit_kwargs=None,
    time_bins_kwargs=None,
    cv=None,
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

    score_func = score_func or get_scores
    fit_kwargs = fit_kwargs or dict()
    time_bins_kwargs = time_bins_kwargs or dict()

    cv_scores = defaultdict(list)

    cv = cv or KFold()
    for train_idxs, val_idxs in cv.split(X):

        if subset:
            n_train, n_val = len(train_idxs), len(val_idxs)

            subset_train = int(subset.get("train", 1) * n_train)
            train_idxs = train_idxs[:subset_train]

            subset_val = int(subset.get("val", 1) * n_val)
            val_idxs = val_idxs[:subset_val]

        print(f"train set: {len(train_idxs)}, val set: {len(val_idxs)}")

        X_train, y_train = X.values[train_idxs, :], y[train_idxs]
        X_val, y_val = X.values[val_idxs, :], y[val_idxs]

        # Generate evaluation time steps as a subset of the train and val durations.
        times = get_times(y_train, y_val)
        # add the right parameter to handle `times`.
        fit_kwargs = _handle_times(estimator, times, time_bins_kwargs, fit_kwargs)

        t0 = perf_counter()
        estimator.fit(X_train, y_train, **fit_kwargs)
        t1 = perf_counter()

        scores = score_func(estimator, y_train, X_val, y_val, times)

        # Accumulate each score into a separate list.
        for k, v in scores.items():
            cv_scores[k].append(v)
        cv_scores["training_duration"].append(t1 - t0)

    # Compute each score mean and std
    score_keys = [
        "ibs",
        "c_index",
        "c_index_ipcw",
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

    save_scores(model_name, cv_scores)


def get_scores(model, y_train, X_test, y_test, times):

    t0 = perf_counter()
    survival_probs = model.predict_survival_function(X_test, return_array=True)
    t1 = perf_counter()

    cumulative_hazards = model.predict_cumulative_hazard_function(
        X_test, return_array=True
    )
    risk_estimate = cumulative_hazards.sum(axis=1)

    _, brier_scores = brier_score(y_train, y_test, survival_probs, times)
    ibs = integrated_brier_score(y_train, y_test, survival_probs, times)

    # The AUC is very close to the IBS in this benchmark, so we don't compute it.
    # auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, cumulative_hazards, times)

    # As the C-index is expensive to compute, we only consider a subsample of our data. 
    N_sample_c_index = 50_000
    c_index = concordance_index_censored(
        y_test["event"][:N_sample_c_index],
        y_test["duration"][:N_sample_c_index],
        risk_estimate[:N_sample_c_index],
    )[0]
    c_index_ipcw = concordance_index_ipcw(
        y_train[:N_sample_c_index],
        y_test[:N_sample_c_index],
        risk_estimate[:N_sample_c_index],
    )[0]

    return dict(
        brier_scores=brier_scores,
        ibs=ibs,
        times=times,
        survival_probs=survival_probs,
        # auc=auc,
        # mean_auc=mean_auc,
        c_index=c_index,
        c_index_ipcw=c_index_ipcw,
        prediction_duration=t1 - t0,
    )


def get_times(y_train, y_val):
    y_time = np.hstack([y_train["duration"], y_val["duration"]])
    lower, upper = np.percentile(y_time, [2.5, 97.5])
    return np.linspace(lower, upper, 100)


def _handle_times(estimator, times, time_bins_kwargs, fit_kwargs):
    """Add the correct `times` parameter to `fit_kwargs`, if any is needed.
    """
    if not isinstance(time_bins_kwargs, dict):
        raise TypeError("`time_bins_kwargs` must be a dict.")
    if estimator.__class__.__name__ == "Pipeline":
        for model_name, arg in time_bins_kwargs.items():
            fit_kwargs[f"{model_name}__{arg}"] = times
        return fit_kwargs

    if len(time_bins_kwargs) > 1:
        raise ValueError(
            "More than 1 element provided in time_bins_kwargs "
            "for non-Pipeline estimator."
        )
    arg = list(time_bins_kwargs.values())[0]
    fit_kwargs[arg] = times

    return fit_kwargs


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
    
    for path in results_dir.iterdir():
        if (
            path.is_file()
            and path.suffix == ".pkl"
            and match_filter in str(path)
        ):
            result = pickle.load(open(path, "rb"))
            model_name = path.name.split(".")[0].split("_")[-1]

            add_lines(result, model_name, lines)
            add_tables(result, model_name, tables)

    df_tables = pd.DataFrame(tables)
    df_lines = pd.DataFrame(lines)
    
    # sort by ibs
    df_tables["ibs_tmp"] = df_tables["IBS"].str.split("±").str[0] \
                                           .astype(np.float64)
    df_tables = df_tables.sort_values("ibs_tmp").reset_index(drop=True)
    df_tables.pop("ibs_tmp")
    
    return df_tables, df_lines
    

def add_lines(result, model_name, lines):
    
    # times are the same across all folds, output shape: (times)
    times = result["times"][0] 
    
    # take the mean for each folds, output shape: (times)
    brier_scores = np.asarray(result["brier_scores"]).mean(axis=0)

    # ensure that all folds has the same size
    N = min([len(el) for el in result["survival_probs"]])
    survival_probs = [el[:N] for el in result["survival_probs"]]
    
    # take the mean for each individual across all folds, output shape: (N, times)
    survival_probs = np.asarray(survival_probs).mean(axis=0)
    
    row = dict(
        model=model_name,
        times=times,
        brier_scores=brier_scores,
        survival_probs=survival_probs,
    )
    lines.append(row)
    

def add_tables(result, model_name, tables):

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
    
    tables.append(row)