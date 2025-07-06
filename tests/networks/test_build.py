# tests/test_build_networks.py

import networkx as nx
import pytest

from apparent.networks.build import NetworkBuilder


def ER_provider_model(edges_df, hsa, year, p=0.4):
    """Model Physician Networks with Erdos-Renyi Graphs"""
    sub_frame = edges_df[(edges_df.hsa == hsa) & (edges_df.year == year)]
    num_providers = len(sub_frame["npi_a"].unique())
    G = nx.erdos_renyi_graph(num_providers, p)
    return G

@pytest.mark.unit
class TestBuildNetwork:
    def test_standard_build(self, edges_df1, edges_df2):
        B1 = NetworkBuilder()

        G1 = B1.standard_build(edges_df1, 1, 2001)

        assert isinstance(G1, nx.Graph)
        assert G1.is_directed() is False
        assert G1.is_multigraph() is False

        G2 = B1.standard_build(edges_df2, 1, 2001)

        assert isinstance(G2, nx.Graph)
        assert G2.is_directed() is False
        assert G2.is_multigraph() is False

    def test_custom_build(self, edges_df1, edges_df2):
        B1 = NetworkBuilder(custom_build=ER_provider_model)

        G1 = B1.build(edges_df1, hsa=1, year=2001)
        assert isinstance(G1, nx.Graph)
        G2 = B1.build(edges_df2, hsa=1, year=2001)
        assert isinstance(G1, nx.Graph)
