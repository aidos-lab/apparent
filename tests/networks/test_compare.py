import pytest
import networkx as nx
import numpy as np
from scott.geometry.measures.ollivier import prob_two_hop
from apparent.networks.compare import NetworkComparator


class TestNetworkComparator:

    def test_defaults(self, cycle_network, path_network):
        """Test initialization with default settings."""
        # Create sample networks
        networks = [cycle_network, path_network]

        # Initialize the comparator
        comparator = NetworkComparator(networks)

        # Validate attributes
        assert isinstance(comparator.networks, list)
        assert len(comparator.networks) == 2
        assert all(isinstance(g, nx.Graph) for g in comparator.networks)

    def test_compare(self, cycle_network, path_network, complete_network):
        """Test the compare function with default settings."""
        networks = [cycle_network, path_network, complete_network]

        # Initialize the comparator
        comparator = NetworkComparator(networks)

        # Compute pairwise distances
        distance_matrix = comparator.compare()

        # Validate output
        assert isinstance(distance_matrix, np.ndarray)
        assert distance_matrix.shape == (3, 3)  # Must be symmetric (3 x 3)
        assert np.allclose(distance_matrix, distance_matrix.T)  # Symmetry check
        assert np.all(
            np.diag(distance_matrix) == 0
        )  # Diagonal must be zero (self-distance)

    def test_compare_with_custom_measure_and_metric(
        self, cycle_network, path_network
    ):
        """Test the compare function with custom measure and metric."""
        networks = [cycle_network, path_network]

        # Initialize the comparator
        comparator = NetworkComparator(networks)

        # Test with a different measure and metric
        distance_matrix = comparator.compare(
            measure="ollivier_ricci_curvature",
            metric="image",
            weight=None,
            alpha=0.5,
            prob_fn=prob_two_hop,
            resolution=[10, 10],  # image specific kwarg
        )

        # Validate output
        assert isinstance(distance_matrix, np.ndarray)
        assert distance_matrix.shape == (2, 2)
        assert np.allclose(distance_matrix, distance_matrix.T)
        assert np.all(np.diag(distance_matrix) == 0)

    def test_invalid_measure(self, cycle_network, path_network):
        """Test handling of invalid curvature measure."""
        # Create sample networks
        networks = [cycle_network, path_network]

        # Initialize the comparator
        comparator = NetworkComparator(networks)

        # Test with an invalid measure
        with pytest.raises(
            AssertionError,
            match="The given curvature measure is not yet supported by KILT.",
        ):
            comparator.compare(measure="invalid_measure")

    def test_invalid_metric(self, cycle_network, path_network):
        """Test handling of invalid comparison metric."""
        # Create sample networks
        networks = [cycle_network, path_network]

        # Initialize the comparator
        comparator = NetworkComparator(networks)

        invalid_metric = "new_persistence_metric"
        # Test with an invalid metric
        with pytest.raises(
            AssertionError, match=f"Metric {invalid_metric} is not supported"
        ):
            comparator.compare(metric=invalid_metric)

    def test_empty_networks(self):
        """Test behavior with an empty network list."""
        with pytest.raises(ValueError, match="No networks to compare."):
            comparator = NetworkComparator(networks=[])
            comparator.compare()

    def test_single_network(self, cycle_network):
        """Test behavior with a single network in the list."""
        comparator = NetworkComparator([cycle_network])

        # Distance matrix should be 1x1 with 0
        distance_matrix = comparator.compare()
        assert distance_matrix.shape == (1, 1)
        assert distance_matrix[0, 0] == 0
