import pytest
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from apparent.curvature import (
    forman_curvature,
    ollivier_ricci_curvature,
    pairwise_resistances,
    node_resistance_curvature,
    resistance_curvature,
)


@pytest.fixture
def test_graphs():
    return [_triangle_graph(), _line_graph(), _tree_graph()]


# Tests
class TestCurvatureFunctions:
    def test_forman_curvature(self, test_graphs):
        tri, line, tree = test_graphs
        # Triangle Graph
        tri_result = forman_curvature(tri)
        tri_expected = np.array([3.0, 3.0, 3.0])
        np.testing.assert_array_equal(tri_result, tri_expected)

        # Line Graph
        line_result = forman_curvature(line)
        line_expected = np.array([0.0] * len(line_result))
        line_expected[0], line_expected[-1] = 1.0, 1.0
        np.testing.assert_array_equal(line_result, line_expected)

        # Tree Graph
        tree_result = forman_curvature(tree)
        tree_expected = [-1.0, 1.0, 0.0, 0.0]
        np.testing.assert_array_equal(tree_result, tree_expected)

    def test_ollivier_ricci_curvature(self, test_graphs):
        tri, line, tree = test_graphs
        # Triangle Graph
        tri_result = ollivier_ricci_curvature(tri, alpha=0.0)
        tri_expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(
            tri_result, tri_expected, decimal=5
        )

        # Line Graph
        line_result = ollivier_ricci_curvature(line, alpha=0.0)
        line_expected = np.array([0.0] * len(line_result))
        np.testing.assert_array_equal(line_result, line_expected)

        # Tree Graph
        tree_result = ollivier_ricci_curvature(tree, alpha=0.0)
        tree_expected = [-1 / 3, 0.0, 0.0, 0.0]
        np.testing.assert_array_almost_equal(
            tree_result, tree_expected, decimal=5
        )

    def test_pairwise_resistances(self, test_graphs):
        tri, line, tree = test_graphs
        # Triangle Graph
        R, node_distances = pairwise_resistances(tri)
        assert R.shape == (3, 3)
        assert node_distances == {1: 0, 2: 1, 3: 2}

        # Line Graph
        R, node_distances = pairwise_resistances(line)
        assert R.shape == (10, 10)
        assert node_distances == {x: x for x in range(10)}

        # Tree Graph
        R, node_distances = pairwise_resistances(tree)
        assert R.shape == (5, 5)
        assert node_distances == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

    def test_node_resistance_curvature(self, test_graphs):
        tri, line, tree = test_graphs

        # Triangle Graph
        tri_results = [
            node_resistance_curvature(
                tri,
                node,
            )
            for node in tri.nodes()
        ]
        tri_expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_almost_equal(tri_results, tri_expected, decimal=5)

        # Line Graph
        line_results = [
            node_resistance_curvature(
                line,
                node,
            )
            for node in line.nodes()
        ]
        line_expected = np.array([0.0] * len(line_results))
        line_expected[0], line_expected[-1] = 0.5, 0.5
        np.testing.assert_array_almost_equal(
            line_results, line_expected, decimal=5
        )

        # Tree Graph
        tree_results = [
            node_resistance_curvature(
                tree,
                node,
            )
            for node in tree.nodes()
        ]
        tree_expected = [0.0, -0.5, 0.5, 0.5, 0.5]
        np.testing.assert_array_almost_equal(
            tree_results, tree_expected, decimal=5
        )

    def test_resistance_curvature(self, test_graphs):
        tri, line, tree = test_graphs
        # Triangle Graph
        tri_result = resistance_curvature(tri)
        tri_expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(
            tri_result, tri_expected, decimal=5
        )

        # Line Graph
        line_result = resistance_curvature(line)
        line_expected = np.array([0.0] * len(line_result))
        line_expected[0], line_expected[-1] = 1.0, 1.0
        np.testing.assert_array_almost_equal(
            line_result, line_expected, decimal=5
        )

        # Tree Graph
        tree_result = resistance_curvature(tree)
        tree_expected = [-1.0, 1.0, 0.0, 0.0]
        np.testing.assert_array_almost_equal(
            tree_result, tree_expected, decimal=5
        )


# Graphs
def _triangle_graph():
    G = nx.Graph()
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 1, weight=1.0)
    return G


def _line_graph():
    return nx.path_graph(10)


def _tree_graph():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])
    assert nx.is_tree(G)
    return G


# Test Arguments
class test_args:
    def __init__(self, curvature, alpha=0, prob_fn=None):
        self.curvature = curvature
        self.alpha = alpha
        self.prob_fn = prob_fn
