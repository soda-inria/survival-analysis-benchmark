import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import mlflow

from sksurv.datasets import get_x_y
from pycox.datasets import kkbox_v1

from models.yasgbt import YASGBTClassifier
from model_selection.cross_validation import run_cv, get_all_results
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
        n_iter=20,
    )
    est = YASGBTClassifier(**est_params)
    est.name = "YASGBT-prod"
    
    # Run cross val
    run_cv(X, y, est)

    # Fetch all scores
    df_tables, df_lines = get_all_results(match_filter=est.name)

    # Save the brier scores figure
    plot_brier_scores(df_lines)
    bs_filename = f"{est.name}_brier_score.png"
    plt.savefig(bs_filename)

    # Save the individuals survival proba curve figure 
    plot_individuals_survival_curve(df_tables, df_lines, y)
    surv_curv_filename = f"{est.name}_surv_curv.png"
    plt.savefig(surv_curv_filename)

    scores = df_tables.to_dict("index")[0]

    # Register metrics and params to mlflow
    print(f"mlflow URI: {mlflow.get_tracking_uri()}")
    with mlflow.start_run(run_name="survival_demo"):
        mlflow.log_metrics(scores)
        mlflow.log_params(est_params)
        mlflow.log_artifact(bs_filename, artifact_path=bs_filename)
        mlflow.log_artifact(surv_curv_filename, artifact_path=surv_curv_filename)


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