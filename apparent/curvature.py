"""Curvature measures for graphs."""

import ot
import warnings

import networkx as nx
import numpy as np


def ollivier_ricci_curvature(G, alpha=0.0, weight=None, prob_fn=None):
    """Calculate Ollivier--Ricci curvature of a graph.

    This function calculates the Ollivier--Ricci curvature of a graph,
    optionally taking (positive) edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    alpha : float
        Provides laziness parameter for default probability measure. The
        measure is not compatible with a user-defined `prob_fn`. If such
        a function is set, `alpha` will be ignored.

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated. Notice that
        if `prob_fn` is provided, this parameter will have no effect for
        the calculation of probability measures, but it will be used for
        the calculation of shortest-path distances.

    prob_fn : callable or None
        If set, should refer to a function that calculate a probability
        measure for a given graph and a given node. This callable needs
        to satisfy the following signature:

        ``prob_fn(G, node, node_to_index)``

        Here, `G` refers to the graph, `node` to the node whose measure
        is to be calculated, and `node_to_index` to the lookup map that
        maps a node identifier to a zero-based index.

        If `prob_fn` is set, providing `alpha` will not have an effect.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    assert 0 <= alpha <= 1

    # Ensures that we can map a node to its index in the graph,
    # regardless of whether node labels or node names are being
    # used.
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    # This is defined inline anyway, so there is no need to follow the
    # same conventions as for the `prob_fn` parameter.
    def _make_probability_measure(node):
        values = np.zeros(len(G.nodes))
        values[node_to_index[node]] = alpha

        degree = G.degree(node, weight=weight)

        for neighbor in G.neighbors(node):

            if weight is not None:
                w = G[node][neighbor][weight]
            else:
                w = 1.0

            values[node_to_index[neighbor]] = (1 - alpha) * w / degree

        return values

    # We pre-calculate all information about the probability measure,
    # making edge calculations easier later on.
    if prob_fn is None:
        measures = list(map(_make_probability_measure, G.nodes))
    else:
        measures = list(map(lambda x: prob_fn(G, x, node_to_index), G.nodes))

    # This is the cost matrix for calculating the Ollivier--Ricci
    # curvature in practice.
    #
    # TODO: Is this the most efficient way?
    M = nx.floyd_warshall_numpy(G, weight=weight)

    curvature = []
    # we get a curvature per edge

    for edge in G.edges():
        source, target = edge

        mi = measures[node_to_index[source]]
        mj = measures[node_to_index[target]]

        distance = ot.emd2(mi, mj, M, numThreads="max")
        curvature.append(1.0 - distance)

    return np.asarray(curvature)


def pairwise_resistances(G, weight=None):
    """Calculate pairwise resistances for all neighbors of a graph.

    Calculate pairwise resistances for all neighbors in a graph `G`
    using the networkx implementation of `resistance_distance`. This
    function helps reducing redundant computations when calculating
    `resistance_curvature`, by doing the necessary calculations up
    front.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated.

    Returns
    -------
    R : np.matrix
        A matrix of pairwise resistance distances between neighboring
        nodes in `G`.

    node_to_index : dict
        A reference dictionary for translating between nodes and indices
        of `G`.
    """
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    n = len(G.nodes())

    # Initialize nxn Matrix
    R = np.zeros(shape=(n, n))

    # List of connected components with original node order
    components = list(
        [G.subgraph(c).copy() for c in nx.connected_components(G)]
    )
    for C in components:
        for source, target in C.edges():
            i, j = node_to_index[source], node_to_index[target]
            r = nx.resistance_distance(
                C,
                source,
                target,
                weight=weight,
                invert_weight=False,
            )
            # Assign Matrix Entries for neighbors
            R[i, j], R[j, i] = r, r

    return R, node_to_index


def node_resistance_curvature(G, node, weight=None, R=None, node_to_index=None):
    """Calculate Resistance Curvature of a given node in a graph 'G'.

    This function calculates the resistance curvature of only
    the nodes in a graph, optionally takes (positive)
    edge weights into account. This is a helper function for
    resistance_curvature; the curvature of each node is used to
    determine the overall curvature of the graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated.

    Returns
    -------
    np.float32
        The node curvature of a given node in `G`.
    """
    assert node in G.nodes()

    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    if R is None:
        R, node_to_index = pairwise_resistances(G, weight=weight)

    neighbors = list(G.neighbors(node))
    rel_resistance = 0

    for neighbor in neighbors:

        if weight is not None and len(G.get_edge_data(node, neighbor)) > 0:
            w = G[node][neighbor][weight]

        else:
            w, G[node][neighbor]["weight"] = 1, 1

        rel_resistance += R[node_to_index[node]][node_to_index[neighbor]] * w

    node_curvature = 1 - 0.5 * rel_resistance

    return np.float32(node_curvature)


def resistance_curvature(G, weight=None):
    """Calculate Resistance Curvature of a graph.

    This function calculates the resistance curvature of a graph,
    optionally taking (positive) edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        # Generate Matrix of Resistance Distances and Node Reference
        # Dictionary
        R, node_to_index = pairwise_resistances(G, weight=weight)
    curvature = []

    for edge in G.edges():
        source, target = edge
        source_curvature = node_resistance_curvature(
            G, source, weight=weight, R=R, node_to_index=node_to_index
        )
        target_curvature = node_resistance_curvature(
            G, target, weight=weight, R=R, node_to_index=node_to_index
        )

        edge_curvature = (
            2
            * (source_curvature + target_curvature)
            / R[node_to_index[source], node_to_index[target]]
        )
        curvature.append(edge_curvature)

    return np.asarray(curvature)


def forman_curvature(G, weight=None):
    """Calculate Forman--Ricci curvature of a graph.

    This function calculates the Forman--Ricci curvature of a graph,
    optionally taking (positive) node and edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. Will use the same attribute to look up node weights. If
        None, unweighted curvature is calculated.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    # This calculation is much more efficient than the weighted one, so
    # we default to it in case there are no weights in the graph.
    if weight is None:
        return _forman_curvature_unweighted(G)
    else:
        return _forman_curvature_weighted(G, weight)


def _forman_curvature_unweighted(G):
    """Calculate Forman--Ricci curvature of an unweighted graph.

    This function calculates the Forman--Ricci curvature of an unweighted graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    curvature = []
    for edge in G.edges():

        source, target = edge
        source_degree = G.degree(source)
        target_degree = G.degree(target)

        source_neighbours = set(G.neighbors(source))
        target_neighbours = set(G.neighbors(target))

        n_triangles = len(source_neighbours.intersection(target_neighbours))
        curvature.append(
            float(4 - source_degree - target_degree + 3 * n_triangles)
        )

    return np.asarray(curvature)


def _forman_curvature_weighted(G, weight):
    """Calculate Forman--Ricci curvature of a weighted graph.

    This function calculates the Forman--Ricci curvature of a weighted graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str
        Name of an edge attribute that is supposed to be used as an edge
        weight.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    has_node_attributes = bool(nx.get_node_attributes(G, weight))

    curvature = []
    for edge in G.edges:
        source, target = edge
        source_weight, target_weight = 1.0, 1.0

        # Makes checking for duplicate edges easier below. We expect the
        # source vertex to be the (lexicographically) smaller one.
        if source > target:
            source, target = target, source

        if has_node_attributes:
            source_weight = G.nodes[source][weight]
            target_weight = G.nodes[target][weight]

        edge_weight = G[source][target][weight]

        e_curvature = source_weight / edge_weight
        e_curvature += target_weight / edge_weight

        parallel_edges = list(G.edges(source, data=weight)) + list(
            G.edges(target, data=weight)
        )

        for u, v, w in parallel_edges:
            if u > v:
                u, v = v, u

            if (u, v) == edge:
                continue
            else:
                e_curvature -= w / np.sqrt(edge_weight * w)

        e_curvature *= edge_weight
        curvature.append(float(e_curvature))

    return np.asarray(curvature)
