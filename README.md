# Benchmarking 

This repository is dedicated to the evaluation of predictive survival models on large-ish datasets.


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


