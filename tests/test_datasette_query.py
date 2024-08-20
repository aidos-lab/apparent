import pytest
import os
import pandas as pd
import networkx as nx
from dotenv import load_dotenv

from apparent.query import fetch_data_from_sql
from apparent.build_networks import build_network
from apparent.curvature import forman_curvature


class TestDatasette:
    def test_local(self):
        load_dotenv()
        D = os.getenv("DATASETTE_URL")
        sql_query = """
          SELECT
            hospital_atlas_data.hsaname,
            hospital_atlas_data.latitude,
            hospital_atlas_data.longitude,
            referral_network_features.forman_mean,
            referral_network_features.forman_median,
            referral_network_features.forman_mode,
            referral_network_features.forman_stdev,
            population_census.race_total_pop,
            population_census.race_black_pop,
            population_census.hispanic_total_pop,
            population_census.hispanic_pop,
            population_census.median_hh_income
          FROM
            hospital_atlas_data
            JOIN referral_network_features ON hospital_atlas_data.hsa = referral_network_features.hsa
            AND hospital_atlas_data.year = referral_network_features.year
            JOIN population_census ON hospital_atlas_data.hsa = population_census.hsa
            AND hospital_atlas_data.year = population_census.year
          WHERE
            hospital_atlas_data.year = 2017;
          """

        edges_df = fetch_data_from_sql(D, sql_query)

        print(edges_df.head())


# df = edges_df[["hsanum", "year"]].drop_duplicates(keep="first")
# df = df.assign(
#     graph=df.swifter.apply(
#         lambda row: build_network(
#             edges_df=edges_df, hsanum=row["hsanum"], year=row["year"]
#         ),
#         axis=1,
#     ),
# )
# df.assign(
#     forman=df.swifter.apply(
#         lambda row: forman_curvature(row["graph"]), axis=1
#     ),
#     axis=1,
# )
