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

## Models

The `models` section define the following models:
- Yet Another Gradient Survival Boosting Tree (YASGBT): wrapper around scikit-learn `HistGradientBoostingTree` optimizing the Brier Score by sampling observation times and recomputing the associated target `y_c` for each iteration.
  ```python
  from models.yasgbt import YASGBTClassifier
  ```
- Kaplan Tree and Kaplan Neighbor: ready to use models whose architecture is
  adapted from [XGBSE](https://loft-br.github.io/xgboost-survival-embeddings/how_xgbse_works.html#xgbsekaplanneighbors-kaplan-meier-on-nearest-neighbors)
  with scikit-learn estimators.
  ```python
  from models.kaplan_tree import KaplanTree
  ```
- Meta GridBC and Tree transformer: wrappers to reproduce the [XGBSEDebiasedBCE](https://loft-br.github.io/xgboost-survival-embeddings/how_xgbse_works.html#xgbsedebiasedbce-logistic-regressions-time-windows-embedding-as-input) architecture.
  ```python
  from sklearn.pipeline import make_pipeline
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestRegressor

  from models.tree_transformer import TreeTransformer
  from models.meta_grid_bc import MetaGridBC
  from model_selection.wrappers import PipelineWrapper

  tree_transformer = TreeTransformer(
      # ignores censoring so it introduces some bias
      # at the cost of speed increase
      RandomForestRegressor()
  )

  meta_grid_bc = MetaGridBC(
      LogisticRegression(),
      verbose=False,
      n_jobs=4,
  )

  forest_grid_bc = make_pipeline(
      tree_transformer,
      meta_grid_bc,
  )

  forest_grid_bc = PipelineWrapper(
      forest_grid_bc,
      name="BiasedForestGridBC"
  )
  ```
