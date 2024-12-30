import pytest
import networkx as nx
import pandas as pd


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
