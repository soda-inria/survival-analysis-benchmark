# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Survival Analysis Tutorial Part 2
#
#
# The goal of this notebook is to extract implicit failure events from a raw activity (or "heart-beat") log using [Ibis](ibis-project.org/) and [DuckDB](https://duckdb.org) or [Polars](https://www.pola.rs/).
#
# It is often the case that the we are dealing with a raw **activity event log** for a pool of members/patients/customers/machines... where the event of interest (e.g. churn, death, hospital transfer, failure) only appears in negative via the lack of activity event for an extended period of time: **activity events are collected somewhat regularly as long as the "failure" event has not occured**.
#
# Our goal is to use a common data-wrangling technique named **sessionization** to infer implicit failure events and measure the duration between the start of activity recording until a failure event (or censoring).
#
# We will also see how censoring naturally occur when we extract time-slices of a sessionized dataset.
#
# Links to the slides:
#
# - https://docs.google.com/presentation/d/1pAFmAFiyTA0_-ZjWG1rImAX8lYJt_UnGgqXD-4H6Aqw/edit?usp=sharing

# %%
import ibis

ibis.options.interactive = True
ibis.__version__

# %%
import duckdb

duckdb.__version__

# %%
from urllib.request import urlretrieve
from pathlib import Path

data_filepath = Path("wowah_data_raw.parquet")
data_url = (
    "https://storage.googleapis.com/ibis-tutorial-data/wowah_data/"
    "wowah_data_raw.parquet"
)

if not data_filepath.exists():
    print(f"Downloading {data_url}...")
    urlretrieve(data_url, data_filepath)
else:
    print(f"Reusing downloaded {data_filepath}")

# %%
conn = ibis.duckdb.connect()  # in-memory DuckDB
event_log = conn.read_parquet(data_filepath)
event_log

# %%
event_log.count()

# %%
from ibis import deferred as c


entity_window = ibis.cumulative_window(
    group_by=c.char, order_by=c.timestamp
)
threshold = ibis.interval(minutes=30)
deadline_date = c.timestamp.lag().over(entity_window) + threshold

(
    event_log
    .select([c.char, c.timestamp])
    .mutate(deadline_date=deadline_date)
)

# %%
(
    event_log
    .select([c.char, c.timestamp])
    .mutate(
        is_new_session=(c.timestamp > deadline_date).fillna(False)
    )
)

# %%
(
    event_log
    .select([c.char, c.timestamp])
    .mutate(
        is_new_session=(c.timestamp > deadline_date).fillna(False)
    )
    .mutate(session_id=c.is_new_session.sum().over(entity_window))
)

# %%
entity_window = ibis.cumulative_window(
    group_by=c.char, order_by=c.timestamp
)
threshold = ibis.interval(minutes=30)
deadline_date = c.timestamp.lag().over(entity_window) + threshold
is_new_session = (c.timestamp > deadline_date).fillna(False)

sessionized = (
    event_log
    .mutate(is_new_session=is_new_session)
    .mutate(session_id=c.is_new_session.sum().over(entity_window))
    .drop("is_new_session")
)
sessions = (
    sessionized
    .group_by([c.char, c.session_id])
    .order_by(c.timestamp)
    .aggregate(
        session_start_date=c.timestamp.min(),
        session_end_date=c.timestamp.max(),
    )
    .order_by([c.char, c.session_start_date])
)
sessions


# %%
# ibis.show_sql(sessions)

# %%
def sessionize(table, threshold, entity_col, date_col):
    entity_window = ibis.cumulative_window(
        group_by=entity_col, order_by=date_col
    )
    deadline_date = date_col.lag().over(entity_window) + threshold
    is_new_session = (date_col > deadline_date).fillna(False)

    return (
        table
        .mutate(is_new_session=is_new_session)
        .mutate(session_id=c.is_new_session.sum().over(entity_window))
        .drop("is_new_session")
    )


def extract_sessions(table, entity_col, date_col, session_col):
    return (
        table
        .group_by([entity_col, session_col])
        .aggregate(
            session_start_date=date_col.min(),
            session_end_date=date_col.max(),
        )
        # XXX: we would like to compute session duration here but
        # it seems broken with Ibis + DuckDB at the moment...
        .order_by([entity_col, c.session_start_date])
    )


def preprocess_event_log(event_log):
    return (
        event_log
        .pipe(
            sessionize,
            threshold=ibis.interval(minutes=30),
            entity_col=c.char,
            date_col=c.timestamp,
        )
        .pipe(
            extract_sessions,
            entity_col=c.char,
            date_col=c.timestamp,
            session_col=c.session_id,
        )
    )


# %%
# %time sessions = preprocess_event_log(event_log).cache()

# %%
# %time sessions.count()

# %%
sessions

# %%
first_observed_date = event_log.timestamp.max().execute()
first_observed_date

# %%
last_observed_date = event_log.timestamp.max().execute()
last_observed_date


# %%
def censor(sessions, censoring_date, threshold=ibis.interval(minutes=30), observation_duration=None):
    if observation_duration is not None:
        sessions = sessions.filter(c.session_start_date > censoring_date - observation_duration)
    return (
        sessions
        .filter(c.session_start_date < censoring_date)
        .mutate(
            is_censored=censoring_date < (c.session_end_date + threshold),
            session_end_date=ibis.ifelse(c.session_end_date < censoring_date, c.session_end_date, censoring_date),
        )
        # remove sessions that are two short
        .filter(c.session_end_date > c.session_start_date + ibis.interval(minutes=1))
        .order_by(c.session_start_date)
    )

censor(sessions, last_observed_date).is_censored.sum()

# %%
from datetime import timedelta

censor(sessions, last_observed_date - timedelta(days=54)).count()

# %%
censor(sessions, last_observed_date - timedelta(days=54), observation_duration=timedelta(days=5)).to_pandas()

# %%
# ibis.show_sql(preprocess_event_log(event_log))

# %%
import polars as pl


pl.__version__

# %%
event_log_df = pl.read_parquet(data_filepath)
event_log_df.head(5)

# %%
event_log_lazy_df = pl.scan_parquet(data_filepath)
event_log_lazy_df.head(10)

# %%
event_log_lazy_df.head(10).collect()


# %%
def sessionize_pl(df, entity_col, date_col, threshold_minutes):
    sessionized = (
        df.sort([entity_col, date_col])
        .with_columns(
            [
                (pl.col(date_col).diff().over(entity_col).dt.minutes() > threshold_minutes)
                .fill_null(False)
                .alias("is_new_session"),
            ]
        )
        .with_columns(
            [
                pl.col("is_new_session").cumsum().over(entity_col).alias("session_id"),
            ]
        )
        .drop(["is_new_session"])
    )
    return sessionized


def extract_sessions_pl(
    df,
    entity_col,
    date_col,
    session_col,
    metadata_cols=["race", "zone", "charclass", "guild"]
):
    sessions = (
        df
        .sort(date_col)
        .groupby([entity_col, session_col])
        .agg(
            [pl.col(mc).first().alias(mc) for mc in metadata_cols]
            + [
                pl.col(date_col).min().alias("session_start_date"),
                pl.col(date_col).max().alias("session_end_date"),
            ]
        )
        .with_columns(
            [
                (pl.col("session_end_date") - pl.col("session_start_date")).alias("session_duration"),
            ]
        )
        .sort([entity_col, "session_start_date"])
    )
    return sessions


def preprocess_event_log_pl(df):
    return (
        df
        .pipe(
            sessionize_pl,
            entity_col="char",
            date_col="timestamp",
            threshold_minutes=30,
        )
        .pipe(
            extract_sessions_pl,
            entity_col="char",
            date_col="timestamp",
            session_col="session_id",
        )
    )


# %time sessions_collected = preprocess_event_log_pl(event_log_lazy_df).collect()
sessions_collected

# %%
first_observed_date = event_log_lazy_df.select("timestamp").min().collect().item()
first_observed_date

# %%
last_observed_date = event_log_lazy_df.select("timestamp").max().collect().item()
last_observed_date


# %%
def censor_pl(sessions, censoring_date, threshold_minutes=30, observation_days=None):
    if observation_days:
        start_date = censoring_date - timedelta(days=observation_days)
        sessions = sessions.filter(pl.col("session_start_date") > start_date)
    return (
        sessions
        .filter(pl.col("session_start_date") < censoring_date)
        .with_columns(
            [
                (((censoring_date - pl.col("session_end_date")).dt.minutes()) < threshold_minutes).alias("is_censored"),
                pl.min(pl.col("session_end_date"), censoring_date).alias("session_end_date"),
            ]
        )
        .with_columns(
            [
                (pl.col("session_end_date") - pl.col("session_start_date")).dt.minutes().alias("duration"),
                (pl.col("is_censored") == False).alias("event"),
            ]
        )
        .filter(pl.col("duration") > 0)
        .sort("session_start_date")
    )


censor_pl(sessions_collected, last_observed_date)

# %%
censor_pl(sessions_collected, last_observed_date).select("is_censored").sum()

# %%
censor_pl(sessions_collected, last_observed_date - timedelta(days=42), observation_days=5)

# %% [markdown]
# ***Wrap-up exercise***
#
# - Select 10 dates randomly from the beginning of January to the end of November (increment the first date with random number of days). For each sample date, define an observation window of 5 days: extract the censored data and concatenate those sessions into a training set;
#
# - Estimate and plot the average survival function using a Kaplan-Meier estimator;
#
# - Reiterate the KM estimation, but stratified on the `race` or the `charclass` features;
#
# - Fit a predictive survival model of your choice with adequate feature engineering on this training set;
#
# - Extract censored data from the last month of the original dataset and use it to measure the performance of your estimator with the metrics of your choice. Compare this to the Kaplan-Meier baseline.
#
# - Inspect which features are the most predictive, one way or another.

# %%
# TODO: write your code here
