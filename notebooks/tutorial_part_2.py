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
# The goal of this tutorial is to extract implicit failure information from a raw event log using [Ibis](ibis-project.org/) and [DuckDB](https://duckdb.org)

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
transactions = conn.read_parquet(data_filepath)
transactions

# %%
transactions.count().execute() / 1e6

# %%
from ibis import deferred as c


entity_window = ibis.cumulative_window(
    group_by=c.char, order_by=c.timestamp
)
threshold = ibis.interval(minutes=30)
deadline_date = c.timestamp.lag().over(entity_window) + threshold

(
    transactions
    .select([c.char, c.timestamp])
    .mutate(deadline_date=deadline_date)
)

# %%
(
    transactions
    .select([c.char, c.timestamp])
    .mutate(
        is_new_session=(c.timestamp > deadline_date).fillna(False)
    )
)

# %%
(
    transactions
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
    transactions
    .mutate(is_new_session=is_new_session)
    .mutate(session_id=c.is_new_session.sum().over(entity_window))
    .drop("is_new_session")
)
sessions = (
    sessionized
    .group_by([c.char, c.session_id])
    .aggregate(
        session_start_date=c.timestamp.min(),
        session_end_date=c.timestamp.max(),
    )
    .order_by([c.char, c.session_start_date])
)
sessions.count().execute() / 1e6


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
        .order_by([entity_col, c.session_start_date])
    )


def preprocess_transactions(transactions):
    return (
        transactions
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
sessions = preprocess_transactions(transactions)
# %time sessions.count().execute() / 1e6

# %%
# %time sessions_df = sessions.to_pandas()
sessions_df

# %%
# ibis.show_sql(preprocess_transactions(transactions))

# %%
import polars as pl


pl.__version__

# %%
transactions_df = pl.read_parquet(data_filepath)
transactions_df.head(5)

# %%
transactions_lazy_df = pl.scan_parquet(data_filepath)
transactions_lazy_df.head(10)

# %%
transactions_lazy_df.head(10).collect()


# %%
def sessionize_pl(df, entity_col, date_col, threshold):
    sessionized = (
        df.sort([entity_col, date_col])
        .with_columns(
            [
                (pl.col(date_col).diff().over(entity_col).dt.minutes() > threshold)
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

def extract_sessions_pl(df, entity_col, date_col, session_col):
    sessions = (
        df.groupby([entity_col, session_col])
        .agg(
            [
                pl.col(date_col).min().alias("session_start_date"),
                pl.col(date_col).max().alias("session_end_date"),
            ]
        )
        .sort([entity_col, "session_start_date"])
    )
    return sessions


def preprocess_transactions_pl(df):
    return (
        df
        .pipe(
            sessionize_pl,
            entity_col="char",
            date_col="timestamp",
            threshold=30,
        )
        .pipe(
            extract_sessions_pl,
            entity_col="char",
            date_col="timestamp",
            session_col="session_id",
        )
    )


# %time sessions_collected = preprocess_transactions_pl(transactions_lazy_df).collect()
sessions_collected

# %%
sessions_collected.shape
