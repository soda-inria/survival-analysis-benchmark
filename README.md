# Benchmarking predictive survival analysis models

This repository is dedicated to the evaluation of predictive survival models on large-ish datasets.

## Software dependencies

To run the jupytercon-2023 tutorial notebooks, you will need:

```
conda create -n jupytercon-survival -c conda-forge python jupyterlab scikit-learn lifelines scikit-survival matplotlib-base plotly seaborn pandas pyarrow ibis-duckdb polars 

conda activate jupytercon-survival
jupyter lab
```

## Notebooks

The `notebooks` folder holds the two main notebooks for the jupytercon-2023, namely:

- `tutorial_part_1.ipynb`
- `tutorial_part_2.ipynb`

and the ancillary notebook used to generate the dataset used in "part 1", namely:

- `truck_dataset.ipynb`

Note that running `truck_dataset.ipynb` consumes a significant share of RAM, so you might prefer [downloading the datasets from zip](https://github.com/soda-inria/survival-analysis-benchmark/releases/download/jupytercon-2023-tutorial-data/truck_failure.zip) (500 MB) instead of generating them.

The notebooks display our benchmark results and show how to use our wrappers to cross validate various models.

- `kkbox_cv_benchmark.ipynb`
  
  Benchmark of the KKBox challenge inspired from the [pycox paper](https://jmlr.org/papers/volume20/18-424/18-424.pdf).
- `msk_mettropism.ipynb`
  
  Exploration of the MSK cancer dataset and survival probability predictions using our models.

## Datasets

### WSDM - KKBox's Churn Prediction Challenge (from Kaggle)

The `datasets/kkbox_churn` folder contains Python code to efficiently
preprocess the raw transaction logs of the KKBox's Churn Prediction Challenge
using [ibis](https://ibis-project.org) and [duckdb](https://duckdb.org).

- https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data

The objectives are to:

- make everything reproducible from the event-based logs;
- implement efficient, parallel and out-of-core "sessionization" of the past
  transactions for all members: here is a "session" is an uninterrupted
  sequence of transactions;
- implement efficient, parallel and out-of-core tabularization (feature
  and churn target with censoring);
- make it possible to compute the cumulative state of the subscription data
  and the censored churn events at any point in time.

Alternatively, `kkbox_cv_benchmark.ipynb` directly uses the preprocessing steps from pycox to ensure reproducibility, at the cost of memory and speed performances.

## Models

This repository introduces a novel survival estimator named Gradient Boosting CIF. This estimator is based on the HistGradientBoostingClassifier of scikit-learn under the hood. 

It is named CIF because it has the capability to estimate cause-specific Cumulative Incidence Functions in a competing risks setting by minimizing a cause specific Integrated Brier Score (IBS) objective function:

```python
from models.gradient_boosted_cif import GradientBoostedCIF

X_train = np.array([
  [5.0, 0.1, 2.0],
  [3.0, 1.1, 2.2],
  [2.0, 0.3, 1.1],
  [4.0, 1.0, 0.9],
])
y_train_multi_event = np.array([
  (2, 33.2),
  (0, 10.1),
  (0, 50.0),
  (1, 20.0),
  ], dtype=[("event", np.bool_), ("duration", np.float64)]
)
time_grid = np.linspace(0.0, 30.0, 10)

gb_cif = GradientBoostedCIF(event_of_interest=1)
gb_cif.fit(X_train, y_train_multi_event, time_grid)

X_test = np.array([[3.0, 1.0, 9.0]])
cif_curves = gb_cif.predict_cumulative_incidence(X_test, time_grid)
```

Alternatively, you can estimate the probability of an event to be experienced at a specific time horizon:

```python
gb_cif.predict_proba(X_test, time_horizon=20)
```

Conversely, you can estimate the conditional quantile time to event e.g. answering the question "at which time horizon does the CIF reach 50%?":

```python
gb_cif.predict_quantile(X_test, quantile=0.5)
```

You can also estimate the survival function in the single event setting (binary event). Warning: this metric only makes sense when `y` is binary or when setting `event_of_interest='any'`.

```python
y_train_single_event = np.array([
  (1, 12.0),
  (0, 5.1),
  (1, 1.1),
  (0, 29.0),
  ], dtype=[("event", np.bool_), ("duration", np.float64)]
) 
gb_cif.fit(X_train, y_train_single_event, time_grid)
survival_curves = gb_cif.predict_survival_function(X_test, time_grid)
```
