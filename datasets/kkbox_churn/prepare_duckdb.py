from time import perf_counter
from pathlib import Path
import ibis
from kkbox_utils import table_schemas


KKBOX_DATA_FOLDER = Path("~/data/kkbox-churn-prediction-challenge").expanduser()

table_name_to_csv_filename = {
    'members': 'members_v3.csv',
}

connection = ibis.duckdb.connect(database='kkbox.db')
existing_tables = connection.list_tables()
for table_name, schema in table_schemas.items():
    if table_name not in existing_tables:
        csv_filename = table_name_to_csv_filename.get(table_name, table_name + '.csv')
        print(f"Loading {csv_filename} to table {table_name}...")
        tic = perf_counter()
        # transaction table
        connection.create_table(table_name, schema=schema)
        connection.raw_sql(
            f"COPY {table_name} FROM '{KKBOX_DATA_FOLDER}/{csv_filename}'"
            f" WITH (FORMAT CSV, HEADER TRUE, DATEFORMAT '%Y%m%d');"
        )
        toc = perf_counter()
        transactions = connection.table(table_name)
        print(
            f"Imported {transactions.count().execute()} rows into "
            f"transactions table in {toc - tic:0.3f} seconds"
        )
    else:
        print(f"Table '{table_name}' already exists")
        transactions = connection.table(table_name)
