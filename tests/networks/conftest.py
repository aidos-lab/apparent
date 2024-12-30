import pytest
import numpy as np
import networkx as nx
import pandas as pd


@pytest.fixture
def empty_network():
    return nx.Graph()


@pytest.fixture
def cycle_network():
    return nx.cycle_graph(4)


@pytest.fixture
def path_network():
    return nx.path_graph(4)


@pytest.fixture
def complete_network():
    return nx.complete_graph(4)


@pytest.fixture
def random_distance_matrix():
    rng = np.random.default_rng()
    matrix = rng.random((10, 10))
    np.fill_diagonal(matrix, 0)
    return matrix


@pytest.fixture
def random_data():
    rng = np.random.default_rng()
    return rng.random((30, 6))


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
