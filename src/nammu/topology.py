"""Representations and calculations of topological features."""

import collections.abc

import networkx as nx
import numpy as np

from nammu.utils import UnionFind


class PersistenceDiagram(collections.abc.Sequence):
    """Persistence diagram class.

    Represents a persistence diagram, i.e. a pairing of nodes in
    a graph. The purpose of this class is to provide a *simpler*
    interface for storing and accessing this pairing.
    """

    def __init__(self, pairs=None, dimension=None):
        """Create new persistence diagram.

        Parameters
        ----------
        pairs:
            Optional sequence of persistence pairs to add to this
            diagram. If set, the diagram will be initialised with
            said sequence. The sequence *must* consist of tuples.

        dimension : int (optional)
            Specifies dimension of the diagram. This can be useful in
            order to denote the tuples that gave rise to a *specific*
            diagram.
        """
        if pairs is None:
            self._pairs = []
        else:
            self._pairs = [(c, d) for c, d in pairs]

        self._dimension = dimension

    def __len__(self):
        """Return the number of pairs in the persistence diagram."""
        return len(self._pairs)

    def __getitem__(self, index):
        """Return the persistence pair at the given index."""
        return self._pairs[index]

    def __truediv__(self, alpha):
        """Elementwise division by a scalar."""
        self._pairs = [(c / alpha, d / alpha) for c, d in self._pairs]
        return self

    def __repr__(self):
        """Return string representation of the diagram."""
        return "\n".join([f"{c}\t{d}" for c, d in self])

    @property
    def dimension(self):
        """Return dimension of the persistence diagram, if set."""
        return self._dimension

    @property
    def persistence(self):
        """Return list of persistence values of this diagram."""
        return [abs(d - c) for c, d in self._pairs]

    @property
    def betti(self):
        """Return Betti number of persistence diagram."""
        return len([(c, d) for c, d in self._pairs if not np.isfinite(d)])

    def add(self, x, y):
        """Append a new persistence pair to the given diagram.

        Extends the persistence diagram by adding a new persistence
        tuple to the diagram. Performs no other validity checks.

        Parameters
        ----------
        x:
            Creation value of the given persistence pair
        y:
            Destruction value of the given persistence pair
        """
        self._pairs.append((x, y))

    def union(self, other):
        """Calculate the union of two persistence diagrams.

        The union of two persistence diagrams is defined as the union of
        their underlying persistence pairs. The current persistence diagram
        is modified in place.

        Parameters
        ----------
        other:
            Other persistence diagram

        Returns
        -------
        Updated persistence diagram.
        """
        for x, y in other:
            self.add(x, y)

        return self

    def total_persistence(self, p=1):
        """Calculate the total persistence of the current pairing.

        The total persistence is closely related to the $p$-norm in that
        it employs a sum of the persistence values found in the diagram.
        In contrast to the norm, though, no additional root will be used
        in this formulation.

        Parameters
        ----------
        p:
            Exponent for the total persistence calculation

        Returns
        -------
        Total persistence with exponent $p$.
        """
        return sum([abs(x - y) ** p for x, y in self._pairs])

    def p_norm(self, p=1):
        """Calculate the $p$-norm of the current pairing.

        Parameters
        ----------
        p : float
            Exponent for the $p$-norm calculation

        Returns
        -------
        $p$-norm of the persistence diagram.
        """
        return sum([abs(x - y) ** p for x, y in self._pairs]) ** (1.0 / p)

    def infinity_norm(self, p=1):
        """Calculate the infinity norm of the current pairing.

        Parameters
        ----------
        p:
            Exponent for the infinity norm calculation

        Returns
        -------
        Infinity norm with exponent $p$.
        """
        return max([abs(x - y) ** p for x, y in self._pairs])


def _has_vertex_attribute(graph, attribute):
    return len(nx.get_node_attributes(graph, attribute)) != 0


def _has_edge_attribute(graph, attribute):
    return len(nx.get_edge_attributes(graph, attribute)) != 0


def calculate_persistence_diagrams(
    graph,
    vertex_attribute="f",
    edge_attribute="f",
    order="sublevel",
    unpaired=None,
):
    """Calculate persistence diagrams for a graph.

    Calculates a set of persistence diagrams for a given graph. The
    graph is already assumed to contain function values on its edge
    and node elements, respectively. Based on this information, the
    function will calculate persistence diagrams using sublevel, or
    superlevel, sets.

    Parameters
    ----------
    graph:
        Input graph. Needs to have vertex and edge attributes for the
        calculation to be valid.

    vertex_attribute:
        Specifies which vertex attribute to use for the calculation of
        persistent homology.

    edge_attribute:
        Specifies with edge attribute to use for the calculation of
        persistent homology.

    order:
        Specifies the filtration order that is to be used for calculating
        persistence diagrams. Can be either 'sublevel' for a sublevel set
        filtration, or 'superlevel' for a superlevel set filtration.

    unpaired : float or `None`
        If set, uses this value to represent unpaired simplices. Else,
        the largest edge weight will be used.

    Returns
    -------
    Set of persistence diagrams, describing topological features of
    a specified dimension.
    """
    n_vertices = graph.number_of_nodes()
    uf = UnionFind(n_vertices)

    # Relabel Nodes as 0-indexed integers
    graph = nx.convert_node_labels_to_integers(graph, first_label=0)

    assert _has_vertex_attribute(graph, vertex_attribute)
    assert _has_edge_attribute(graph, edge_attribute)

    # The edge weights will be sorted according to the pre-defined
    # filtration that has been specified by the client.
    edge_weights = np.asarray(
        list(nx.get_edge_attributes(graph, edge_attribute).values()),
        dtype=object,
    )
    edge_indices = None

    # Ditto for the vertex weights and indices---with the difference
    # that we will require the `vertex_indices` array to look up the
    # position of a vertex with respect to the given filtration. In
    # other words, `vertex_indices[i]` points towards the *rank* of
    # vertex i.
    vertex_weights = np.asarray(
        list(nx.get_node_attributes(graph, vertex_attribute).values())
    )
    vertex_indices = np.empty_like(vertex_weights)

    # Will contain all the edges that are responsible for cycles in the
    # graph.
    edge_indices_cycles = []

    assert order in ["sublevel", "superlevel"]

    if order == "sublevel":
        edge_indices = np.argsort(edge_weights, kind="stable")

        # Required to ensure that the array can be used as a look-up
        # table. See above for more discussion.
        vertex_indices[np.argsort(vertex_weights, kind="stable")] = np.arange(
            len(vertex_weights)
        )

    # Like the professional that I am, we just need to flip the edge
    # weights here. Note that we do not make *any* assumptions about
    # whether this is consistent with respect to the nodes. The same
    # goes for the vertex indices, by the weight.
    else:
        edge_indices = np.argsort(-edge_weights, kind="stable")

        # Required to ensure that the array can be used as a look-up
        # table. See above for more discussion.
        vertex_indices[np.argsort(-vertex_weights, kind="stable")] = np.arange(
            len(vertex_weights)
        )

    # TODO: is it possible to check vertex and edge indices for
    # consistency? Only if one uses a running index between all
    # of the *simplices* in the graph. In this case, every edge
    # has to be preceded by its vertices.

    # Will be filled during the iteration below. This will become
    # the return value of the function.
    persistence_diagram_0 = PersistenceDiagram()

    edges = list(graph.edges)
    vertex_attributes = nx.get_node_attributes(graph, vertex_attribute)

    # Go over all edges and optionally create new points for the
    # persistence diagram. This is the main loop for our current
    # filtration, and merely requires keeping track of connected
    # component information.
    for edge_index, edge_weight in zip(
        edge_indices, edge_weights[edge_indices]
    ):
        u, v = edges[edge_index]

        # Preliminary assignment of younger and older component. We
        # will check below whether this is actually correct, for it
        # is possible that `u` is actually the older one.
        younger = uf.find(u)
        older = uf.find(v)

        # Nothing to do here: the two components are already the
        # same, so the edge gives rise to a *cycle*.
        if younger == older:
            edge_indices_cycles.append(edge_index)
            continue

        # Ensures that the older component *precedes* the younger one
        # in terms of its vertex index. In other words its index must
        # be *smaller* than that of the younger component. This might
        # sound counterintuitive, but is correct.
        elif vertex_indices[younger] < vertex_indices[older]:
            u, v = v, u
            younger, older = older, younger

        vertex_weight = vertex_attributes[younger]

        creation = vertex_weight  # x coordinate for persistence diagram
        destruction = edge_weight  # y coordinate for persistence diagram

        # Merge the *younger* connected component into the older one,
        # thus preserving the 'elder rule'.
        uf.merge(u, v)
        persistence_diagram_0.add(creation, destruction)

    # By default, use the largest weight to assign to unpaired
    # vertices. This is consistent with *extended persistence*
    # calculations.
    if len(edge_weights) > 0:
        unpaired_value = edge_weights[edge_indices[-1]]
    else:
        unpaired_value = 0

    # Use the user-provided value if available.
    if unpaired is not None:
        unpaired_value = unpaired

    # Add tuples for every root component in the Union--Find data
    # structure. This ensures that multiple connected components
    # are handled correctly.
    for root in uf.roots():

        vertex_weight = vertex_attributes[root]

        creation = vertex_weight
        destruction = unpaired_value

        persistence_diagram_0.add(creation, destruction)

    # Create a persistence diagram for the cycles in the data set.
    # Notice that these are *not* properly destroyed; a better, or
    # smarter, calculation would be warranted.
    persistence_diagram_1 = PersistenceDiagram()

    for edge_index in edge_indices_cycles:
        persistence_diagram_1.add(edge_weights[edge_index], unpaired_value)

    return persistence_diagram_0, persistence_diagram_1
