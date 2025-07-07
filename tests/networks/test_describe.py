import pytest
import networkx as nx
from scott.kilt import CURVATURE_MEASURES
from apparent.networks.describe import NetworkDescriber

@pytest.mark.unit
class TestNetworkDescriber:

    def test_defaults(self, empty_network):
        """Test initialization with default settings."""
        describer = NetworkDescriber(empty_network)

        assert isinstance(describer.G, nx.Graph)
        assert describer.features == {}

    def test_supported_metrics(self, cycle_network):
        """Test that only supported metrics raise no errors."""
        describer = NetworkDescriber(cycle_network)

        # Node features
        supported_node_features = [
            "degree",
            "clustering",
            "betweenness",
            "closeness",
            "pagerank",
        ]
        describer.compute_node_features(supported_node_features)

        for feature in supported_node_features:
            assert feature in describer.features

        # Edge features
        supported_edge_features = ["edge_betweenness"] + list(
            CURVATURE_MEASURES
        )
        describer.compute_edge_features(supported_edge_features)

        for feature in supported_edge_features:
            assert feature in describer.features

    def test_compute_node_features(self, cycle_network):
        """Test computation of specific node features."""
        describer = NetworkDescriber(cycle_network)

        # Compute degree and clustering
        describer.compute_node_features(["degree", "clustering"])

        # Check if features are added
        assert "degree" in describer.features
        assert "clustering" in describer.features

        # Validate degree values
        degrees = dict(cycle_network.degree())
        assert describer.features["degree"] == degrees

        # Validate clustering values
        clustering = nx.clustering(cycle_network)
        assert describer.features["clustering"] == clustering

    def test_compute_edge_features(self, cycle_network):
        """Test computation of specific edge features."""
        describer = NetworkDescriber(cycle_network)

        # Compute edge betweenness
        describer.compute_edge_features(["edge_betweenness"])

        # Check if feature is added
        assert "edge_betweenness" in describer.features

        # Validate edge betweenness values
        edge_betweenness = nx.edge_betweenness_centrality(cycle_network)
        assert describer.features["edge_betweenness"] == edge_betweenness

    def test_compute_features(self, cycle_network):
        """Test computation of both node and edge features."""
        describer = NetworkDescriber(cycle_network)

        # Compute both node and edge features
        describer.compute_features(
            node_features=["degree", "clustering"],
            edge_features=["edge_betweenness"],
        )

        # Validate computed features
        assert "degree" in describer.features
        assert "clustering" in describer.features
        assert "edge_betweenness" in describer.features

    def test_get_features(self, cycle_network):
        """Test retrieval of computed features."""
        describer = NetworkDescriber(cycle_network)

        # Compute features
        describer.compute_features(
            node_features=["degree"], edge_features=["edge_betweenness"]
        )

        features = describer.get_features()

        # Validate output
        assert isinstance(features, dict)
        assert "degree" in features
        assert "edge_betweenness" in features

    def test_unknown_node_feature(self, cycle_network):
        """Test error handling for unknown node features."""
        describer = NetworkDescriber(cycle_network)

        with pytest.raises(
            ValueError, match="Unknown node feature: invalid_feature"
        ):
            describer.compute_node_features(["invalid_feature"])

    def test_unknown_edge_feature(self, cycle_network):
        """Test error handling for unknown edge features."""
        describer = NetworkDescriber(cycle_network)

        with pytest.raises(
            ValueError, match="Unknown edge feature: invalid_feature"
        ):
            describer.compute_edge_features(["invalid_feature"])
