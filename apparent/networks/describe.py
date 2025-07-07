import networkx as nx
from scott.kilt import KILT, CURVATURE_MEASURES


class NetworkDescriber:
    """
    A class to compute and describe features for a given graph.

    This class provides functionality to compute various node and edge features
    for NetworkX graphs. It supports standard network centrality measures as well
    as curvature-based edge features through the Scott package.

    Parameters
    ----------
    G : nx.Graph
        Input NetworkX graph to analyze.

    Attributes
    ----------
    G : nx.Graph
        Input graph with updated node and edge attributes after feature computation.
    features : dict
        Dictionary containing computed features with feature names as keys.

    Examples
    --------
    >>> import networkx as nx
    >>> from apparent.networks import NetworkDescriber
    >>> 
    >>> # Create a sample graph
    >>> G = nx.karate_club_graph()
    >>> 
    >>> # Compute features
    >>> describer = NetworkDescriber(G)
    >>> describer.compute_features(
    ...     node_features=['degree', 'betweenness'],
    ...     edge_features=['forman_curvature']
    ... )
    >>> print(f"Computed features: {list(describer.features.keys())}")
    """

    def __init__(self, G):
        """
        Initialize the NetworkDescriber with a graph.

        Parameters
        ----------
        G : nx.Graph
            Input NetworkX graph to analyze. The graph will be modified in-place
            when features are computed.

        Examples
        --------
        >>> G = nx.karate_club_graph()
        >>> describer = NetworkDescriber(G)
        """
        self.G = G
        self.features = {}

    def compute_node_features(self, node_features):
        """
        Compute specified node features for the graph and add them as attributes.

        This method computes various centrality and structural measures for nodes
        and adds them as node attributes to the graph.

        Parameters
        ----------
        node_features : list
            List of node features to compute. Available options:
            - 'degree': Node degree (number of connections)
            - 'clustering': Local clustering coefficient
            - 'betweenness': Betweenness centrality
            - 'closeness': Closeness centrality
            - 'pagerank': PageRank centrality

        Returns
        -------
        None
            Features are added as node attributes to self.G and stored in self.features.

        Raises
        ------
        ValueError
            If an unknown node feature is specified.

        Examples
        --------
        >>> describer = NetworkDescriber(G)
        >>> describer.compute_node_features(['degree', 'betweenness'])
        >>> print(G.nodes[0]['degree'])  # Access computed degree
        """
        for feature in node_features:
            if feature == "degree":
                values = dict(self.G.degree())
            elif feature == "clustering":
                values = nx.clustering(self.G)
            elif feature == "betweenness":
                values = nx.betweenness_centrality(self.G)
            elif feature == "closeness":
                values = nx.closeness_centrality(self.G)
            elif feature == "pagerank":
                values = nx.pagerank(self.G)
            else:
                raise ValueError(f"Unknown node feature: {feature}")

            # Add feature to graph nodes
            nx.set_node_attributes(self.G, values, name=feature)
            self.features[feature] = values

    def compute_edge_features(self, edge_features, **kwargs):
        """
        Compute specified edge features for the graph and add them as attributes.

        This method computes various edge-level features, including centrality measures
        and curvature-based features through the Scott package's KILT implementation.

        Parameters
        ----------
        edge_features : list
            List of edge features to compute. Available options:
            - 'edge_betweenness': Edge betweenness centrality
            - Any curvature measure from scott.kilt.CURVATURE_MEASURES
              (e.g., 'forman_curvature', 'ollivier_ricci_curvature')
        **kwargs : dict
            Additional arguments passed to KILT curvature measures.

        Returns
        -------
        None
            Features are added as edge attributes to self.G and stored in self.features.

        Raises
        ------
        ValueError
            If an unknown edge feature is specified.

        Examples
        --------
        >>> describer = NetworkDescriber(G)
        >>> describer.compute_edge_features(['edge_betweenness', 'forman_curvature'])
        >>> print(G.edges[(0, 1)]['forman_curvature'])  # Access computed curvature
        """
        for feature in edge_features:
            if feature == "edge_betweenness":
                values = nx.edge_betweenness_centrality(self.G)
            elif feature in CURVATURE_MEASURES:
                kilt = KILT(measure=feature, **kwargs)
                kilt.fit(self.G)
                values = dict(zip(self.G.edges(), kilt.curvature))
            else:
                raise ValueError(f"Unknown edge feature: {feature}")

            # Add feature to graph edges
            nx.set_edge_attributes(self.G, values, name=feature)
            self.features[feature] = values

    def compute_features(
        self, node_features=None, edge_features=None, **kwargs
    ):
        """
        Compute all specified features for the graph and add them as attributes.

        This is a convenience method that calls both compute_node_features and
        compute_edge_features based on the provided feature lists.

        Parameters
        ----------
        node_features : list, optional
            List of node features to compute. If None, no node features are computed.
        edge_features : list, optional
            List of edge features to compute. If None, no edge features are computed.
        **kwargs : dict
            Additional arguments passed to KILT curvature measures.

        Returns
        -------
        None
            Features are added as attributes to self.G and stored in self.features.

        Examples
        --------
        >>> describer = NetworkDescriber(G)
        >>> describer.compute_features(
        ...     node_features=['degree', 'clustering'],
        ...     edge_features=['forman_curvature']
        ... )
        """
        if node_features:
            self.compute_node_features(node_features)
        if edge_features:
            self.compute_edge_features(edge_features, **kwargs)

    def get_features(self):
        """
        Retrieve computed features.

        Returns
        -------
        dict
            Dictionary containing computed features with feature names as keys
            and feature values as values.

        Examples
        --------
        >>> describer = NetworkDescriber(G)
        >>> describer.compute_features(node_features=['degree'])
        >>> features = describer.get_features()
        >>> print(features.keys())  # ['degree']
        """
        return self.features
