# %%
import numpy as np
import ibis
from ibis import _ as 𓅠  # to avoid name confict with Jupyter's _


duckdb_conn = ibis.duckdb.connect(database="kkbox.db", read_only=True)
transactions = duckdb_conn.table("transactions")

# %%
def add_resubscription_window(
    transactions,
    expire_threshold=ibis.interval(days=30),
    transaction_threshold=ibis.interval(days=60),
):
    """Flag transactions that are resubscriptions.

    Compute the numbers of days elapsed between a given transaction and a
    deadline computed from the informations in the previous transaction by the
    same member.

    The new resubscription deadline is set to 30 days after the expiration date
    of the previous transaction or 60 days after the date of the previous
    transaction, which ever is the latest. We need this double criterion
    because some expiration dates seems invalid (too short) and would cause the
    detection of too many registration events, even for users which have more
    than one transaction per month for several contiguous months.

    Note: this strategy to detect resubscriptions is simplistic because it does
    not handle cancellation events.
    """
    w = ibis.trailing_window(
        group_by=𓅠.msno,
        order_by=[𓅠.transaction_date, 𓅠.membership_expire_date],
        preceding=1,
    )
    previous_deadline_date = ibis.greatest(
        𓅠.membership_expire_date.first().over(w) + expire_threshold,
        𓅠.transaction_date.first().over(w) + transaction_threshold,
    )
    current_transaction_date = 𓅠.transaction_date.last().over(w)
    resubscription = current_transaction_date > previous_deadline_date
    return transactions.mutate(
        elapsed_days_since_previous_deadline=ibis.greatest(
            current_transaction_date - previous_deadline_date, 0
        ),
        resubscription=resubscription,
    )


#%%


def add_resubscription_groupby_lag(
    transactions,
    expire_threshold=ibis.interval(days=30),
    transaction_threshold=ibis.interval(days=60),
):
    """Flag transactions that are resubscriptions.

    Compute the numbers of days elapsed between a given transaction and a
    deadline computed from the informations in the previous transaction by the
    same member.

    The new resubscription deadline is set to 30 days after the expiration date
    of the previous transaction or 60 days after the date of the previous
    transaction, which ever is the latest. We need this double criterion
    because some expiration dates seems invalid (too short) and would cause the
    detection of too many registration events, even for users which have more
    than one transaction per month for several contiguous months.

    Note: this strategy to detect resubscriptions is simplistic because it does
    not handle cancellation events.
    """
    t = transactions
    return (
        t.group_by(t.msno)
        .order_by([t.transaction_date, t.membership_expire_date])
        # XXX: we have to assign the lag variable as expression fields
        # otherwise we can get a CatalogException from duckdb...
        # Similar constraints for
        .mutate(
            transaction_lag=𓅠.transaction_date.lag(),
            expire_lag=𓅠.membership_expire_date.lag(),
        )
        .mutate(
            transaction_deadline=ibis.coalesce(𓅠.transaction_lag, 𓅠.transaction_date)
            + transaction_threshold,
            expire_deadline=ibis.coalesce(𓅠.expire_lag, 𓅠.membership_expire_date)
            + expire_threshold,
        )
        .mutate(
            previous_deadline_date=ibis.greatest(
                𓅠.transaction_deadline,
                𓅠.expire_deadline,
            )
        )
        .mutate(
            elapsed_days_since_previous_deadline=ibis.greatest(
                𓅠.transaction_date - 𓅠.previous_deadline_date, 0
            ),
            resubscription=t.transaction_date > 𓅠.previous_deadline_date,
        )
        .drop(
            "transaction_lag",
            "expire_lag",
            "transaction_deadline",
            "expire_deadline",
            "previous_deadline_date",
        )
    )


# %%
example_msno = "+8ZA0rcIhautWvUbAM58/4jZUvhNA4tWMZKhPFdfquQ="
(
    transactions.filter(𓅠.msno == example_msno)
    .pipe(add_resubscription_groupby_lag)
    .order_by([𓅠.msno, 𓅠.transaction_date, 𓅠.membership_expire_date])
).execute()

# %%
(
    transactions.filter(𓅠.msno == example_msno)
    .pipe(add_resubscription_window)
    .order_by([𓅠.msno, 𓅠.transaction_date, 𓅠.membership_expire_date])
).execute()

# %%
def count_resubscriptions(expr):
    return expr.group_by(expr.msno).aggregate(
        n_resubscriptions=expr.resubscription.sum(),
    )


# %%
counts_groupby_lag = (
    transactions.pipe(add_resubscription_groupby_lag)
    .pipe(count_resubscriptions)
    .group_by("n_resubscriptions")
    .aggregate(
        n_members=𓅠.msno.count(),
    )
).execute()
counts_groupby_lag

# %%
counts_window = (
    transactions.pipe(add_resubscription_window)
    .pipe(count_resubscriptions)
    .group_by("n_resubscriptions")
    .aggregate(
        n_members=𓅠.msno.count(),
    )
).execute()
counts_window

# %%
assert (counts_window == counts_groupby_lag).all().all()

# %%
# Both methods return the same results on duckdb and take the same
# time. From now one use the group_by + lag variant since not
# all the backends support the generic Window Function API
# add_resubscription = add_resubscription_window
add_resubscription = add_resubscription_groupby_lag


# %%
(
    transactions.pipe(add_resubscription)
    .pipe(count_resubscriptions)
    .order_by([𓅠.n_resubscriptions.desc(), 𓅠.msno])
    .limit(10)
).execute()

# %%
def add_n_resubscriptions(expr):
    return expr.group_by(expr.msno).mutate(
        n_resubscriptions=𓅠.resubscription.sum(),
    )


# %%
example_msno = "AHDfgFvwL4roCSwVdCbzjUfgUuibJHeMMl2Nx0UDdjI="
(
    transactions.filter(𓅠.msno == example_msno)
    .pipe(add_resubscription)
    .pipe(add_n_resubscriptions)
    .order_by([𓅠.transaction_date, 𓅠.membership_expire_date])
).execute()


# %%
def add_subscription_id_window(
    expr, relative_to_epoch=False, epoch=ibis.date(2000, 1, 1)
):
    """Generate a distinct id for each subscription.

    The subscription id is based on the cumulative sum of past resubscription
    events.
    """
    w = ibis.window(
        group_by=𓅠.msno,
        order_by=[𓅠.transaction_date, 𓅠.membership_expire_date],
        preceding=None,
        following=0,
    )
    if relative_to_epoch:
        # use oldest transaction date as reference to make it possible
        # to generate a session id that can be computed in parallel
        # on partitions of the transaction logs.
        base = (𓅠.transaction_date.first().over(w) - epoch).cast("string")
        counter = 𓅠.resubscription.sum().over(w).cast("string")
        subscription_id = base + "_" + counter
    else:
        subscription_id = 𓅠.resubscription.sum().over(w).cast("string")
    return expr.mutate(
        subscription_id=subscription_id,
    )


# %%
def add_subscription_groupby_cumsum(expr):
    """Generate a distinct id for each subscription.

    The subscription id is based on the cumulative sum of past resubscription
    events.
    """
    return (
        expr.group_by(𓅠.msno)
        .order_by([𓅠.transaction_date, 𓅠.membership_expire_date])
        .mutate(
            subscription_id=𓅠.resubscription.cast("int").cumsum().cast("string"),
        )
    )


# %%
add_subscription_id = add_subscription_groupby_cumsum

# %%
example_msno = "AHDfgFvwL4roCSwVdCbzjUfgUuibJHeMMl2Nx0UDdjI="
(
    transactions.filter(𓅠.msno == example_msno)
    .pipe(add_resubscription)
    .pipe(add_n_resubscriptions)
    .pipe(add_subscription_id)
    .order_by([𓅠.transaction_date, 𓅠.membership_expire_date])
).execute()

# %%
def subsample_by_unique(expr, col_name="msno", size=1, seed=None):
    unique_col = expr[[col_name]].distinct()
    num_unique = unique_col.count().execute()
    assert size <= num_unique
    positional_values = unique_col.order_by(col_name)[
        ibis.row_number().name("seq_id"), col_name
    ]
    selected_indices = np.random.RandomState(seed).choice(
        num_unique, size=size, replace=False
    )
    selected_rows = positional_values.filter(
        positional_values.seq_id.isin(selected_indices)
    )[[col_name]]
    return expr.inner_join(selected_rows, col_name, suffixes=["", "_"]).select(expr)


# %%
(
    transactions.pipe(subsample_by_unique, "msno", size=3, seed=0)
    .pipe(add_resubscription)
    .pipe(add_n_resubscriptions)
    .pipe(add_subscription_id)
    .order_by([𓅠.msno, 𓅠.transaction_date, 𓅠.membership_expire_date])
).execute()

# %%
(
    transactions.pipe(subsample_by_unique, "msno", size=3, seed=0)
    .pipe(add_resubscription)
    .pipe(add_n_resubscriptions)
    .pipe(add_subscription_id)
    .order_by([𓅠.msno, 𓅠.transaction_date, 𓅠.membership_expire_date])
).execute()


# %%
from time import perf_counter


def bench_sessionization(conn):
    tic = perf_counter()
    results = (
        (
            conn.table("transactions")
            .pipe(add_resubscription)
            .pipe(add_n_resubscriptions)
            .pipe(add_subscription_id)
            .order_by(
                [
                    ibis.desc("transaction_date"),
                    ibis.desc("membership_expire_date"),
                    "msno",
                ]
            )
            .select("msno", "subscription_id", "n_resubscriptions", "transaction_date")
        )
        .limit(10)
        .execute()
    )
    toc = perf_counter()
    print(f"Sessionization took {toc - tic:.1f} seconds")
    print(results)


bench_sessionization(duckdb_conn)

# %%
parquet_files = {"transactions": "transactions.parquet"}

# %%
duckdb_parquet_conn = ibis.duckdb.connect()
for table_name, path in parquet_files.items():
    duckdb_parquet_conn.register(path, table_name)

bench_sessionization(duckdb_parquet_conn)

# %%
# XXX: pandas window functions are not trustworthy
# ValueError: Can only compare identically-labeled Series objects
# might or not be related to:
# https://github.com/ibis-project/ibis/issues/4676

# The lag / cumsum variants should be supported though, but it is
# slow as there is no parallel computation with the pandas backend...

# import pandas as pd

# pandas_conn = ibis.pandas.connect(
#     {k: pd.read_parquet(v) for k, v in parquet_files.items()}
# )
# bench_sessionization(pandas_conn)

# %%
# XXX: dask does not support window functions
# NotImplementedError: Window operations are unsupported in the dask backend
# XXX: even the lag / cumsum variant raise NotImplementedError with dask

# import dask.dataframe as dd
# dask_conn = ibis.dask.connect(
#     {k: dd.read_parquet(v) for k, v in parquet_files.items()}
# )
# bench_sessionization(dask_conn)

# %% XXX: ibis' Arrow DataFusion backend does not translate Window ops, neight
# with the generic Window API nor with the lag / cumsum variants:
# OperationNotDefinedError: No translation rule for <class
# 'ibis.expr.operations.analytic.Window'>
# datafusion_conn = ibis.datafusion.connect(parquet_files)
# bench_sessionization(datafusion_conn)

# %%
# XXX: polars does not support window functions
# OperationNotDefinedError: No translation rule for <class 'ibis.expr.operations.analytic.Window'>

# XXX: even for the lag / cumsum variant does not work...
# polars_conn = ibis.polars.connect()
# for table_name, path in parquet_files.items():
#     polars_conn.register_parquet(name=table_name, path=path)

# bench_sessionization(polars_conn)

# %%
# Note: to use clickhouse, one needs to first start the server with `clickhouse server`.

# XXX: the following raises NotImplementedError..., too bad.
# ibis.clickhouse.connect().create_table("transactions", pd.read_parquet("transactions.parquet"))

# %%
# clickhouse_conn.raw_sql("DROP TABLE transactions")
# CREATE_QUERY = """\
# CREATE TABLE transactions
# (
#     `msno` String,
#     `payment_method_id` Int8,
#     `payment_plan_days` Int16,
#     `plan_list_price` Int16,
#     `actual_amount_paid` Int16,
#     `is_auto_renew` Int8,
#     `transaction_date` Date,
#     `membership_expire_date` Date,
#     `is_cancel` Int8
# )
# ENGINE = MergeTree
# PARTITION BY toYYYYMM(transaction_date)
# ORDER BY (msno, transaction_date, membership_expire_date)
# """
# clickhouse_conn.raw_sql(CREATE_QUERY)
# !cat "transactions.parquet" | clickhouse client --query="INSERT INTO transactions FORMAT Parquet"
# %%
clickhouse_conn = ibis.clickhouse.connect()
bench_sessionization(clickhouse_conn)

# %%
