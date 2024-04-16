# tests/test_build_networks.py

import pandas as pd
import networkx as nx
import pytest

from apparent.build_networks import build_network


@pytest.fixture
def edges_df1():
    return pd.DataFrame(
        {
            "hsanum": [1, 1, 1, 2],
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
            "hsanum": [
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


class TestBuildNetwork:
    def test_build_network(self, edges_df1, edges_df2):
        G1 = build_network(edges_df1, 1, 2001)
        assert isinstance(G1, nx.Graph)
        assert G1.number_of_nodes() == 4
        assert G1.number_of_edges() == 3
        assert G1.is_directed() is False
        assert G1.is_multigraph() is False

        G2 = build_network(edges_df2, 1, 2001)
        assert isinstance(G2, nx.Graph)
        assert G2.number_of_nodes() == 2
        assert G2.number_of_edges() == 1
        assert G2.is_directed() is False
        assert G2.is_multigraph() is False
