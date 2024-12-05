# tests/test_build_networks.py

import networkx as nx
import pytest

from apparent.networks.build import NetworkBuilder


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
