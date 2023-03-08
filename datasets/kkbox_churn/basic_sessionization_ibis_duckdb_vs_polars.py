# %%
import ibis
from ibis import deferred as c

ibis.options.interactive = True

duckdb_on_disk_conn = ibis.duckdb.connect(database="kkbox.db")
transactions = duckdb_on_disk_conn.table("transactions")
transactions

# %%
transactions.count()


# %%
entity_window = ibis.cumulative_window(
    group_by=c.msno, order_by=c.transaction_date
)
threshold = ibis.interval(days=60)
deadline_date = c.transaction_date.lag().over(entity_window) + threshold

(
    transactions
    .select([c.msno, c.transaction_date])
    .mutate(deadline_date=deadline_date)
)


# %%
(
    transactions
    .select([c.msno, c.transaction_date])
    .mutate(
        is_new_session=(c.transaction_date > deadline_date).fillna(False)
    )
)

# %%
(
    transactions
    .select([c.msno, c.transaction_date])
    .mutate(
        is_new_session=(c.transaction_date > deadline_date).fillna(False)
    )
    .mutate(session_id=c.is_new_session.sum().over(entity_window))
)


# %%
entity_window = ibis.cumulative_window(
    group_by=c.msno, order_by=c.transaction_date
)
threshold = ibis.interval(days=60)
deadline_date = c.transaction_date.lag().over(entity_window) + threshold
is_new_session = (c.transaction_date > deadline_date).fillna(False)

sessionized = (
    transactions.mutate(is_new_session=is_new_session)
    .mutate(session_id=c.is_new_session.sum().over(entity_window))
    .drop("is_new_session")
)
sessions = (
    sessionized.group_by([c.msno, c.session_id])
    .aggregate(
        session_start_date=c.transaction_date.min(),
        session_end_date=c.transaction_date.max(),
    )
    .order_by([c.msno, c.session_start_date])
)

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
            threshold=ibis.interval(days=60),
            entity_col=c.msno,
            date_col=c.transaction_date,
        )
        .pipe(
            extract_sessions,
            entity_col=c.msno,
            date_col=c.transaction_date,
            session_col=c.session_id,
        )
    )

# %%
%time preprocess_transactions(transactions).count().execute()

# %%
duckdb_in_mem_conn = ibis.duckdb.connect()
transactions = duckdb_in_mem_conn.read_parquet("transactions.parquet")
%time preprocess_transactions(transactions).count().execute()

# %%
clickhouse_conn = ibis.clickhouse.connect(host="localhost", port=9000)
transactions = clickhouse_conn.table("transactions")
%time preprocess_transactions(transactions).count().execute()

# %%
# ibis.show_sql(preprocess_transactions(transactions))

# %%
# user_logs = duckdb_on_disk_conn.table("user_logs")
# def preprocess_user_logs(user_logs):
#     return (
#         user_logs
#         .pipe(sessionize, threshold=ibis.interval(days=2), entity_col=c.msno, date_col=c.date)
#         .pipe(extract_sessions, entity_col=c.msno, date_col=c.date, session_col=c.session_id)
#     )
# %time preprocess_user_logs(user_logs).count().execute()

# %%
import polars as pl

df = pl.read_parquet("transactions.parquet")
df

# %%
lazy_df = pl.scan_parquet("transactions.parquet")
lazy_df

# %%
lazy_df.head(10).collect()


# %%
sessionized = (
    lazy_df.sort(["msno", "transaction_date"])
    .with_columns(
        [
            (pl.col("transaction_date").diff().over("msno").dt.days() > 60)
            .fill_null(False)
            .alias("is_new_session"),
        ]
    )
    .with_columns(
        [
            pl.col("is_new_session").cumsum().over("msno").alias("session_id"),
        ]
    )
    .drop(["is_new_session"])
)
sessions = (
    sessionized.groupby(["msno", "session_id"])
    .agg(
        [
            pl.col("transaction_date").min().alias("session_start_date"),
            pl.col("transaction_date").max().alias("session_end_date"),
        ]
    )
    .sort(["msno", "session_start_date"])
)
%time sessions_collected = sessions.collect()
sessions_collected

# %%
sessions_collected.shape
# %%
