# %%
from time import perf_counter
import ibis
# from kkbox_utils import table_schemas


connection = ibis.duckdb.connect(database="kkbox.db")
transactions = connection.table("transactions")
members = connection.table("members")
user_logs = connection.table("user_logs")

parquet_files = {"transactions": "transactions.parquet"}
# connection = ibis.datafusion.connect(parquet_files)
connection = ibis.dask.connect(parquet_files)
transactions = connection.table("transactions")

# %%
def get_transactions_for(t, msno):
    return t.filter(t.msno == msno).sort_by(
        [t.transaction_date, t.membership_expire_date]
    )


example_msno = "+8ZA0rcIhautWvUbAM58/4jZUvhNA4tWMZKhPFdfquQ="
example_transactions = get_transactions_for(transactions, example_msno)
example_transactions.execute()

# %%
def compute_resubscriptions(t):
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
    w = ibis.window(
        group_by=t.msno,
        order_by=[t.transaction_date, t.membership_expire_date],
        preceding=1,
        following=0,
    )
    previous_deadline_date = ibis.greatest(
        t.membership_expire_date.first().over(w) + ibis.interval(days=30),
        t.transaction_date.first().over(w) + ibis.interval(days=60),
    )
    current_transaction_date = t.transaction_date.last().over(w)
    resubscription = current_transaction_date > previous_deadline_date
    return t.mutate(
        elapsed_days_since_previous_deadline=ibis.greatest(
            current_transaction_date - previous_deadline_date, 0
        ),
        resubscription=resubscription,
    )


# %%
compute_resubscriptions(example_transactions).execute()

# %%
# Check that we do not change the number of rows by mistake
assert (
    compute_resubscriptions(transactions).count().execute()
    == transactions.count().execute()
)

# %%
r = compute_resubscriptions(transactions)
resubscription_count = r.group_by(r.msno).aggregate(
    resubscription_count=r.resubscription.sum(),
)

resubscription_count.group_by("resubscription_count").aggregate(
    member_count=resubscription_count.msno.count(),
).execute()

# %%
resubscription_count.sort_by(
    ibis.desc(resubscription_count.resubscription_count)
).limit(10).execute()

# %%
resubscription_count.filter(resubscription_count.resubscription_count == 4).limit(
    10
).execute()

# %%
example_msno = "+8ZA0rcIhautWvUbAM58/4jZUvhNA4tWMZKhPFdfquQ="
t = transactions[transactions.msno == example_msno]
r = compute_resubscriptions(t)
r.sort_by(r.transaction_date).execute()

# %%
def compute_first_transaction_date(t):
    return t.group_by("msno").aggregate(
        reg_init_date=t.transaction_date.min(),
    )


first_t = compute_first_transaction_date(t)
t.left_join(first_t, t.msno == first_t.msno).execute()

# %%
def compute_current_subscription_init_date(t):
    w = ibis.window(
        group_by="msno", order_by="transaction_date", preceding=1, following=0
    )
    current_subscription_init_date = t.resubscription.ifelse(
        t.transaction_date.last().over(w),
        ibis.NA,
    )
    t = t.mutate(current_subscription_init_date=current_subscription_init_date)
    return t


compute_current_subscription_init_date(compute_resubscriptions(t)).execute()

# %%
def transactions_for_member(t, msno):
    return t.filter(t.msno == msno).sort_by(
        [t.transaction_date, t.membership_expire_date]
    )


# %%
transactions_for_member(
    transactions, "Z5bM5ILk0gU0IA/5ys+47rAFOahH55sRtE6oLfMo4q4="
).execute()

# %%
# TODO: investigate why the following is broken
# r = compute_resubscriptions(transactions)
# transactions_for_member(r, "Z5bM5ILk0gU0IA/5ys+47rAFOahH55sRtE6oLfMo4q4=").execute()

# %%
def compute_subscription_id(t, relative_to_epoch=False, epoch=ibis.date(2000, 1, 1)):
    """Generate a distinct id for each subscription.

    The subscription id is based on the cumulative sum of past resubscription
    events.
    """
    w = ibis.window(
        group_by=t.msno,
        order_by=[t.transaction_date, t.membership_expire_date],
        preceding=None,  # all previous transactions
        following=0,
    )
    if relative_to_epoch:
        # use oldest transaction date as reference to make it possible
        # to generate a session id that can be computed in parallel
        # on partitions of the transaction logs.
        base = (t.transaction_date.first().over(w) - epoch).cast("string")
        counter = t.resubscription.sum().over(w).cast("string")
        subscription_id = base + "_" + counter
    else:
        subscription_id = t.resubscription.sum().over(w).cast("string")
    return t.mutate(
        subscription_id=subscription_id,
    )


def compute_subscriptions(t, relative_to_epoch=False, epoch=ibis.date(2000, 1, 1)):
    return compute_subscription_id(
        compute_resubscriptions(t), relative_to_epoch=relative_to_epoch, epoch=epoch
    )


compute_subscriptions(example_transactions).execute()

# %%
compute_subscriptions(example_transactions, relative_to_epoch=True).execute()

# %%
assert (
    compute_subscription_id(compute_resubscriptions(transactions)).count().execute()
    == transactions.count().execute()
)
# %%
import numpy as np


def subsample_by_unique(t, col_name="msno", size=1, seed=None):
    unique_col = t[[col_name]].distinct()
    num_unique = unique_col.count().execute()
    assert size <= num_unique
    positional_values = unique_col[ibis.row_number().name("seq_id"), col_name]
    selected_indices = np.random.RandomState(seed).choice(
        num_unique, size=size, replace=False
    )
    selected_rows = positional_values.filter(
        positional_values.seq_id.isin(selected_indices)
    )[[col_name]]
    return t.inner_join(selected_rows, col_name, suffixes=["", "_"])[t.columns]


subsample_by_unique(transactions, "msno", size=3).sort_by(
    ["msno", "transaction_date", "membership_expire_date"]
).execute()

# %%

compute_subscriptions(subsample_by_unique(transactions, "msno", size=3)).sort_by(
    ["msno", "transaction_date", "membership_expire_date"]
).execute()

# %%
