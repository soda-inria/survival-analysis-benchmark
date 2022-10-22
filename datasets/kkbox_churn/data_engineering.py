# %%
import numpy as np
import ibis
from ibis import _ as 𓅠  # to avoid name confict with Jupyter's _

# import dask.dataframe as dd

connection = ibis.duckdb.connect(database="kkbox.db", read_only=True)

# parquet_files = {"transactions": "transactions.parquet"}
# connection = ibis.dask.connect(
#     {k: dd.read_parquet(v) for k, v in parquet_files.items()}
# )

transactions = connection.table("transactions")
# members = connection.table("members")
# user_logs = connection.table("user_logs")


# %%
def add_resubscription(
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


# %%
def count_resubscriptions(expr):
    return expr.group_by(𓅠.msno).aggregate(
        n_resubscriptions=𓅠.resubscription.sum(),
    )


(
    transactions.pipe(add_resubscription)
    .pipe(count_resubscriptions)
    .group_by("n_resubscriptions")
    .aggregate(
        n_members=𓅠.msno.count(),
    )
).execute()

# %%
(
    transactions.pipe(add_resubscription)
    .pipe(count_resubscriptions)
    # XXX: does not work with ibis.desc(𓅠.n_resubscriptions)
    .order_by([ibis.desc("n_resubscriptions"), "msno"])
    .limit(10)
).execute()

# %%
def add_n_resubscriptions(expr):
    counted = expr.pipe(count_resubscriptions)
    return expr.left_join(counted, expr.msno == counted.msno).select(
        expr, counted.n_resubscriptions
    )


# %%
example_msno = "+8ZA0rcIhautWvUbAM58/4jZUvhNA4tWMZKhPFdfquQ="
(
    transactions.filter(𓅠.msno == example_msno)
    .pipe(add_resubscription)
    .pipe(add_n_resubscriptions)
    .order_by([𓅠.transaction_date, 𓅠.membership_expire_date])
).execute()


# %%
def add_subscription_id(expr, relative_to_epoch=False, epoch=ibis.date(2000, 1, 1)):
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
    .pipe(add_subscription_id)
    .order_by([𓅠.msno, 𓅠.transaction_date, 𓅠.membership_expire_date])
).execute()

# %%
(
    transactions.pipe(subsample_by_unique, "msno", size=3, seed=0)
    .pipe(add_resubscription)
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
