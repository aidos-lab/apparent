import urllib.parse
import pandas as pd


def fetch_data_from_sql(base_url, sql_query):
    try:
        # Encode the SQL query
        encoded_query = urllib.parse.quote(sql_query)

        # Construct the full URL
        url = f"{base_url}?sql={encoded_query}"

        # Fetch data using pandas
        df = pd.read_csv(url)

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
