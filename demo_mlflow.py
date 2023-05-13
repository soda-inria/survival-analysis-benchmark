"""
This file is a quick demo to leverage model versioning with MLFlow for production purposes.
"""
import os
import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature

from sksurv.datasets import get_x_y
from pycox.datasets import kkbox_v1

from models.yasgbt import YASGBTClassifier
from model_selection.cross_validation import (
    run_cv, get_all_results, get_time_grid
)
from plot.brier_score import plot_brier_scores
from plot.individuals import plot_individuals_survival_curve


assert os.getenv("MLFLOW_S3_ENDPOINT_URL"), "env variable MLFLOW_S3_ENDPOINT_URL must be set"
assert os.getenv("PYCOX_DATA_DIR"), "env variable PYCOX_DATA_DIR must be set"

def main():

    # Download data if necessary and run preprocessing
    download_data()
    X, y = preprocess()

    # Define our model
    est_params = dict(
        sampling_strategy="uniform",
        n_iter=10,
    )
    est = YASGBTClassifier(**est_params)
    est.name = "YASGBT-prod"
    
    # Run cross val
    run_cv(X, y, est, single_fold=True)

    # Fetch all scores
    df_tables, df_lines = get_all_results(match_filter=est.name)

    # Get the brier scores figure
    fig_bs = plot_brier_scores(df_lines)
    bs_filename = f"{est.name}_brier_score.png"

    # Get the individuals survival proba curve figure 
    fig_indiv = plot_individuals_survival_curve(df_tables, df_lines, y)
    surv_curv_filename = f"{est.name}_surv_curv.png"

    scores = df_tables.to_dict("index")[0]
    metrics = dict(
        mean_ibs=float(scores["IBS"].split("±")[0]),
        mean_c_index=float(scores["C_td"].split("±")[0]),
    )

    # Get our model signature for mlflow UI
    times = get_time_grid(y, y, n=100)
    X_sample = X.head()
    surv_probs = est.predict_survival_function(X_sample.values, times)
    signature = infer_signature(X_sample, surv_probs)

    # Register metrics and params to mlflow
    print(f"mlflow URI: {mlflow.get_tracking_uri()}")
    with mlflow.start_run(run_name="survival_demo"):
        mlflow.log_metrics(metrics)
        mlflow.log_params(est_params)
        mlflow.log_figure(fig_bs, artifact_file=bs_filename)
        mlflow.log_figure(fig_indiv, artifact_file=surv_curv_filename)
        mlflow.sklearn.log_model(
            est,
            artifact_path=est.name,
            signature=signature,
        )


def download_data():

    kkbox_v1._path_dir.mkdir(exist_ok=True)

    train_file = kkbox_v1._path_dir / "train.csv"
    members_file = kkbox_v1._path_dir / "members_v3.csv"
    transactions_file = kkbox_v1._path_dir / "transactions.csv"

    any_prior_file_missing = (
        not train_file.exists()
        or not members_file.exists()
        or not transactions_file.exists()
    )

    covariate_file = kkbox_v1._path_dir / "covariates.feather"
    is_covariate_file_missing = not covariate_file.exists()

    if is_covariate_file_missing:
        print("Covariate file missing!")
        # We need to download any missing prior file
        # before producing the final covariate file.
        if any_prior_file_missing:
            print("Prior files missing!")
            kkbox_v1._setup_download_dir()
            kkbox_v1._7z_from_kaggle()

        kkbox_v1._csv_to_feather_with_types()
        kkbox_v1._make_survival_data()
        kkbox_v1._make_survival_covariates()
        kkbox_v1._make_train_test_split()


def preprocess():
    covariates = pd.read_feather(kkbox_v1._path_dir / "covariates.feather")
    covariates = extra_cleaning(covariates)
    X, y = get_x_y(covariates, ("event", "duration"), pos_label=1)
    return X, y


def extra_cleaning(df):
    # remove id
    df.pop("msno")

    # ordinal encode gender
    df["gender"] = df["gender"].astype(str)
    gender_map = dict(
        zip(df["gender"].unique(), range(df["gender"].nunique()))
    )
    df["gender"] = df["gender"].map(gender_map)

    # remove tricky np.nan in city, encoded as int
    df["city"] = df["city"].astype(str).replace("nan", -1).astype(int)

    # same for registered via
    df["registered_via"] = (
        df["registered_via"]
        .astype(str)
        .replace("nan", -1)
        .astype(int)
    )

    return df


if __name__ == "__main__":
    main()