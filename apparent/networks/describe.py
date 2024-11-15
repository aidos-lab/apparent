import networkx as nx

# KILT
import sys

sys.path.append("/Users/jeremy.wayland/repos/dev/curvature-filtrations")
from curvature_filtrations.kilt import KILT, CURVATURE_MEASURES


class NetworkDescriber:

    # TODO: Curvature, NetworkX, Cycle Representatives with OatPython
    pass


# Compute Features
def compute_features(G, node_features, edge_features, **kwargs):

    # TODO: Need to figure out how to pass curvature specific measures to KILT
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

        ## CURVATURE
        elif feature in CURVATURE_MEASURES:
            # TODO: CHECK kwargs and pass proper ones to KILT
            kilt = KILT(measure=feature, **kwargs)
            kilt.fit(G)
            # Probably a nice getter from nx to get edge/feature dict
            features[feature] = dict(zip(G.edges(), kilt.curvature))
        else:
            raise ValueError(f"Unknown edge feature: {feature}")

    return features
