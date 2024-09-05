# Build Graphs
def build_network(edges_df, hsanum, year):
    """Create a networkx graph given a dataframe of edges, a region, and year."""

    # initialize an undirected graph G
    G = nx.Graph()

    # populate G
    G = nx.from_pandas_edgelist(
        df=edges_df[(edges_df.hsanum == hsanum) & (edges_df.year == year)],
        source="npi_a",
        target="npi_b",
        edge_attr=["a2b", "b2a"],
    )

    # sanity check
    assert G.is_directed() is False
    assert G.is_multigraph() is False

    return G


def process_row(row):
    A, nodes, curvature = build_network(edges_df, row["hsanum"], row["year"])
    return pd.Series({"adjacency": A, "nodes": nodes, "curvature": curvature})


# Compute Features
def compute_features(G, node_features, edge_features):
    """Compute features for a given graph.

    This function computes features for a given graph. The features
    are computed based on the node and edge attributes of the graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    node_features : list
        List of node features to compute

    edge_features : list
        List of edge features to compute

    Returns
    -------
    features : dict
        Dictionary containing computed features
    """
    features = {}

    # Compute node features
    for feature in node_features:
        if feature == "degree":
            features[feature] = dict(G.degree())
        elif feature == "clustering":
            features[feature] = nx.clustering(G)
        elif feature == "betweenness":
            features[feature] = nx.betweenness_centrality(G)
        elif feature == "closeness":
            features[feature] = nx.closeness_centrality(G)
        elif feature == "pagerank":
            features[feature] = nx.pagerank(G)
        else:
            raise ValueError(f"Unknown node feature: {feature}")

    # Compute edge features
    for feature in edge_features:
        if feature == "edge_betweenness":
            features[feature] = nx.edge_betweenness_centrality(G)
        else:
            raise ValueError(f"Unknown edge feature: {feature}")

    return features
