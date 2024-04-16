import pytest
import networkx as nx
import numpy as np


from apparent.utils import propagate_node_attribute_to_edges
from apparent.topology import calculate_persistence_diagrams


class TestConnecteComponents:
    def test_er_graph(self):
        for p in [0.01, 0.5, 0.9]:
            G = nx.erdos_renyi_graph(100, p)
            n_components = nx.number_connected_components(G)

            degrees = dict(nx.degree(G))
            nx.set_node_attributes(G, degrees, "degree")

            propagate_node_attribute_to_edges(G, "degree")

            diagrams = calculate_persistence_diagrams(
                G, "degree", "degree", unpaired=np.inf
            )

            diagram = diagrams[0]
            assert diagram.betti == n_components


class TestCycles:
    def test_cycle_graph(self):
        for n in [10, 100, 1000]:
            G = nx.cycle_graph(n)

            degrees = dict(nx.degree(G))
            nx.set_node_attributes(G, degrees, "degree")

            propagate_node_attribute_to_edges(G, "degree")

            diagrams = calculate_persistence_diagrams(
                G, "degree", "degree", unpaired=np.inf
            )

            assert diagrams[0].betti == 1

            # All the tuples must be the same since there is no
            # different creation time than destruction time.
            for c, d in diagrams[0]:
                if np.isfinite(d):
                    assert c == d

            # We know precisely how the structure of this graph looks;
            # there must only be a single non-trivial feature.
            assert len(diagrams[1]) == 1
            assert diagrams[1].betti == 1
            assert diagrams[1][0] == (2, np.inf)

    def two_cyle_graph(self):

        G1 = nx.cycle_graph(10)
        G2 = nx.cycle_graph(6)

        F = nx.compose(G1, G2)

        degrees = dict(nx.degree(F))
        nx.set_node_attributes(F, degrees, "degree")

        propagate_node_attribute_to_edges(F, "degree")

        diagrams = calculate_persistence_diagrams(
            F, "degree", "degree", unpaired=np.inf
        )
        print()
        assert diagrams[1].betti == 2

    def three_cyle_graph(self):

        G1 = nx.cycle_graph(10)
        G2 = nx.cycle_graph(6)
        G3 = nx.cycle_graph(3)

        F = nx.compose(G1, G2)
        F = nx.compose(F, G3)

        degrees = dict(nx.degree(F))
        nx.set_node_attributes(F, degrees, "degree")

        propagate_node_attribute_to_edges(F, "degree")

        diagrams = calculate_persistence_diagrams(
            F, "degree", "degree", unpaired=np.inf
        )
        assert diagrams[1].betti == 2
