import duckdb
from pathlib import Path


DATA_FOLDER = Path("~/data/kkbox-churn-prediction-challenge").expanduser()

members_dtypes = {
    "msno": "varchar",
    "city": "int32",
    "bd": "int32",
    "gender": "varchar",
    "registered_via": "int32",
    "registration_init_time": "date",
}

transactions_dtypes = {
    "msno": "varchar",
    "payment_method_id": "int32",
    "payment_plan_days": "int32",
    "plan_list_price": "int32",
    "actual_amount_paid": "int32",
    "is_auto_renew": "boolean",
    "transaction_date": "date",
    "membership_expire_date": "date",
    "is_cancel": "boolean",
}

user_logs = {
    "msno": "varchar",
    "date": "date",
    "num_25": "int32",
    "num_50": "int32",
    "num_75": "int32",
    "num_985": "int32",
    "num_100": "int32",
    "num_unq": "int32",
    "total_secs": "double",
}

tables = {
    "members": {
        "csv_filename": "members_v3.csv",
        "parquet_filename": "members.parquet",
        "dtype": members_dtypes,
    },
    "transactions": {
        "csv_filename": "transactions.csv",
        "parquet_filename": "transactions.parquet",
        "dtype": transactions_dtypes,
    },
    "user_logs": {
        "csv_filename": "user_logs.csv",
        "parquet_filename": "user_logs.parquet",
        "dtype": user_logs,
    },
}

conn = duckdb.connect()

for table_name, table in tables.items():
    csv_file = str(DATA_FOLDER / table["csv_filename"])
    if Path(table["parquet_filename"]).exists():
        print(f"Skipping {csv_file}...")
        continue

    print(f"Processing {csv_file}...")
    df = conn.read_csv(
        csv_file,
        header=True,
        dtype=table["dtype"],
        date_format="%Y%m%d",
    )
    df.to_parquet(table["parquet_filename"])
    print(f"Done writing {table['parquet_filename']}")
