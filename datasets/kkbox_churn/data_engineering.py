import numpy as np
import ibis
from ibis import _ as c  # to avoid name confict with Jupyter's _


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
    t = transactions
    return (
        t.group_by(t.msno)
        .order_by([t.transaction_date, t.membership_expire_date])
        # XXX: we have to assign the lag variable as expression fields
        # otherwise we can get a CatalogException from duckdb... There are
        # similar constraints for subsequent operations in the chained calls to
        # mutate.
        #
        # TODO: craft a minimal reproducible example and report it to the ibis
        # issue tracker.
        .mutate(
            transaction_lag=c.transaction_date.lag(),
            expire_lag=c.membership_expire_date.lag(),
        )
        .mutate(
            transaction_deadline=ibis.coalesce(c.transaction_lag, c.transaction_date)
            + transaction_threshold,
            expire_deadline=ibis.coalesce(c.expire_lag, c.membership_expire_date)
            + expire_threshold,
        )
        .mutate(
            previous_deadline_date=ibis.greatest(
                c.transaction_deadline,
                c.expire_deadline,
            )
        )
        .mutate(
            elapsed_days_since_previous_deadline=ibis.greatest(
                c.transaction_date - c.previous_deadline_date, 0
            ),
            resubscription=t.transaction_date > c.previous_deadline_date,
        )
        .drop(
            "transaction_lag",
            "expire_lag",
            "transaction_deadline",
            "expire_deadline",
            "previous_deadline_date",
        )
    )


def add_subscription_id(expr):
    """Generate a distinct id for each subscription.

    The subscription id is based on the cumulative sum of past resubscription
    events.
    """
    return (
        expr.group_by(c.msno)
        .order_by([c.transaction_date, c.membership_expire_date])
        .mutate(
            subscription_id=c.resubscription.cast("int").cumsum().cast("string"),
        )
    )


if __name__ == "__main__":
    from pathlib import Path

    database = Path(__file__).parent / "kkbox.db"
    duckdb_conn = ibis.duckdb.connect(database=database, read_only=True)
    transactions = duckdb_conn.table("transactions")
    example_msno = "AHDfgFvwL4roCSwVdCbzjUfgUuibJHeMMl2Nx0UDdjI="
    df = (
        transactions.filter(c.msno == example_msno)
        .pipe(add_resubscription)
        .pipe(add_subscription_id)
        .order_by([c.transaction_date, c.membership_expire_date])
    ).execute()
    print(df)