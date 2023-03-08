from time import perf_counter
from pathlib import Path
import ibis


KKBOX_DATA_FOLDER = Path("~/data/kkbox-churn-prediction-challenge").expanduser()

table_name_to_csv_filename = {
    "members": "members_v3.csv",
}

table_schemas = {
    "members": ibis.Schema.from_tuples(
        [
            ("msno", "string"),
            ("city", "int32"),
            ("bd", "int32"),
            ("gender", "string"),
            ("registered_via", "int32"),
            ("registration_init_time", "date"),
        ]
    ),
    "transactions": ibis.Schema.from_tuples(
        [
            ("msno", "string"),
            ("payment_method_id", "int32"),
            ("payment_plan_days", "int32"),
            ("plan_list_price", "int32"),
            ("actual_amount_paid", "int32"),
            ("is_auto_renew", "boolean"),
            ("transaction_date", "date"),
            ("membership_expire_date", "date"),
            ("is_cancel", "boolean"),
        ]
    ),
    "user_logs": ibis.Schema.from_tuples(
        [
            ("msno", "string"),
            ("date", "date"),
            ("num_25", "int32"),
            ("num_50", "int32"),
            ("num_75", "int32"),
            ("num_985", "int32"),
            ("num_100", "int32"),
            ("num_unq", "int32"),
            ("total_secs", "float64"),
        ]
    ),
}

connection = ibis.duckdb.connect(database="kkbox.db")
existing_tables = connection.list_tables()
for table_name, schema in table_schemas.items():
    if table_name not in existing_tables:
        csv_filename = table_name_to_csv_filename.get(table_name, table_name + ".csv")
        print(f"Loading {csv_filename} to table {table_name}...")
        tic = perf_counter()
        # transaction table
        connection.create_table(table_name, schema=schema)
        # XXX: use ibis.duckdb.from_csv() instead?
        connection.raw_sql(
            f"COPY {table_name} FROM '{KKBOX_DATA_FOLDER}/{csv_filename}'"
            f" WITH (FORMAT CSV, HEADER TRUE, DATEFORMAT '%Y%m%d');"
        )
        toc = perf_counter()
        table = connection.table(table_name)
        print(
            f"Imported {table.count().execute()} rows into "
            f"table '{table_name}' in {toc - tic:0.3f} seconds"
        )
    else:
        print(f"Table '{table_name}' already exists")
        table = connection.table(table_name)
    parquet_filename = table_name + ".parquet"
    if not Path(parquet_filename).exists():
        print(f"Writing {table_name} to {parquet_filename}...")
        tic = perf_counter()
        table.to_parquet(parquet_filename)
        toc = perf_counter()
        print(f"Wrote {table_name} to {parquet_filename} in {toc - tic:0.3f} seconds")
