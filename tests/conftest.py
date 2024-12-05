import pytest
import os
import pandas as pd
from dotenv import load_dotenv


@pytest.fixture
def local_url():
    load_dotenv()
    return os.getenv("LOCAL_URL")


@pytest.fixture
def apparent_url():
    load_dotenv()
    return os.getenv("APPARENT_URL")


@pytest.fixture
def max_HSA():
    """Largest HSA in the dataset (2017)"""
    return 45131


@pytest.fixture
def max_HSA_YEAR():
    return 2017


@pytest.fixture
def max_HSA_size():
    return 136354


@pytest.fixture
def max_HSA_query(max_HSA, max_HSA_YEAR):
    return f"SELECT * FROM local_physician_interactions WHERE hsa = {max_HSA} AND year = {max_HSA_YEAR};"


@pytest.fixture
def sample_query():
    sql_query = """
          SELECT
            hospital_atlas_data.hsa,
            hospital_atlas_data.year,
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
            hospital_atlas_data.year = 2017
          LIMIT
            5;
          """
    return sql_query


@pytest.fixture
def edges_df1():
    return pd.DataFrame(
        {
            "hsa": [1, 1, 1, 2],
            "year": [2001, 2001, 2001, 2002],
            "npi_a": [
                "a1",
                "a1",
                "a2",
                "a2",
            ],  # Two Physicians a1,a2
            "npi_b": [
                "b1",
                "b2",
                "b1",
                "b2",
            ],  # Referrerals to other physicians b1,b2
            "a2b": [10, 0, 3, 15],  # number of patients from a to b
            "b2a": [0, 3, 0, 5],  # number of patients from b to a
        }
    )


@pytest.fixture
def edges_df2():
    return pd.DataFrame(
        {
            "hsa": [
                1,
                1,
            ],
            "year": [
                2001,
                2001,
            ],
            "npi_a": [
                "a1",
                "a2",
            ],  # Two Physicians a1,a2
            "npi_b": [
                "a2",
                "a1",
            ],  # Referrerals to other physicians b1,b2
            "a2b": [10, 15],  # number of patients from a to b
            "b2a": [15, 10],  # number of patients from b to a
        }
    )
