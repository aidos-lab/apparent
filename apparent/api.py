"Query and interact with our US Physician Referral Network Datasette."
import polars as pl
import argparse
from utils import fetch_data


def read_query(file_path):
    with open(file_path, "r") as file:
        return file.read()


def fetch_data_from_sql(base_url, sql_query):
    try:
        # Encode the SQL query
        encoded_query = urllib.parse.quote(sql_query)

        # Construct the full URL
        url = f"{base_url}?sql={encoded_query}"

        # Fetch data using pandas
        df = pl.read_csv(url)

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
