import networkx as nx
from scott.kilt import KILT, CURVATURE_MEASURES


class NetworkDescriber:
    """
    A class to compute and describe features for a given graph.

    Attributes
    ----------
    G : nx.Graph
        Input graph with updated node and edge attributes
    features : dict
        Dictionary containing computed features
    """

    def __init__(self, G):
        """
        Initialize the NetworkDescriber with a graph.

        Parameters
        ----------
        G : nx.Graph
            Input graph
        """
        self.G = G
        self.features = {}

    def compute_node_features(self, node_features):
        """
        Compute specified node features for the graph and add them as attributes.

        Parameters
        ----------
        node_features : list
            List of node features to compute

        Returns
        -------
        None
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

        Parameters
        ----------
        edge_features : list
            List of edge features to compute

        **kwargs
            Additional arguments for KILT curvature measures

        Returns
        -------
        None
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

        Parameters
        ----------
        node_features : list, optional
            List of node features to compute

        edge_features : list, optional
            List of edge features to compute

        **kwargs
            Additional arguments for KILT curvature measures

        Returns
        -------
        None
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
            Dictionary containing computed features
        """
        return self.features
