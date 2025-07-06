import pytest
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from apparent.networks.cluster import NetworkClusterer

@pytest.mark.unit
class TestNetworkClusterer:

    def test_defaults(self):
        """Test default initialization."""
        clusterer = NetworkClusterer()
        assert isinstance(clusterer.model, AgglomerativeClustering)
        assert clusterer.model.linkage == "average"
        assert clusterer.model.n_clusters == 2
        assert clusterer.labels_ is None

    def test_standard_fit(self):
        """Test fitting with a standard symmetric pairwise distance matrix."""
        distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        clusterer = NetworkClusterer(clusterer="agglomerative", n_clusters=2)
        labels = clusterer.fit(pairwise_distances=distances)

        assert labels is not None
        assert len(labels) == 3

    def test_manual_labels(self, manual_cluster_labels):
        """Test providing manual labels instead of fitting."""
        clusterer = NetworkClusterer()
        labels = clusterer.fit(manual_labels=manual_cluster_labels)

        assert np.array_equal(labels, manual_cluster_labels)
        assert clusterer.labels_ is not None

    def test_precomputed_distances(self, random_distance_matrix):
        """Test clustering with precomputed distances."""

        clusterer = NetworkClusterer(
            clusterer=AgglomerativeClustering(
                n_clusters=2, metric="precomputed", linkage="average"
            )
        )
        labels = clusterer.fit(pairwise_distances=random_distance_matrix)

        assert labels is not None
        assert len(labels) == len(random_distance_matrix)

        clusterer = NetworkClusterer(
            clusterer="kmeans",
            n_clusters=2,
        )
        labels = clusterer.fit(pairwise_distances=random_distance_matrix)

        assert labels is not None
        assert len(labels) == len(random_distance_matrix)

    def test_kwargs(self):
        """Test passing additional kwargs to the clustering model."""
        clusterer = NetworkClusterer(
            clusterer="kmeans", n_clusters=2, n_init=10, max_iter=100
        )

        assert isinstance(clusterer.model, KMeans)
        assert clusterer.model.n_clusters == 2
        assert clusterer.model.n_init == 10
        assert clusterer.model.max_iter == 100

        clusterer = NetworkClusterer(clusterer="dbscan", eps=0.5, min_samples=5)
        assert isinstance(clusterer.model, DBSCAN)
        assert clusterer.model.eps == 0.5
        assert clusterer.model.min_samples == 5

    def test_invalid_kwargs(self):
        with pytest.raises(
            ValueError,
            match="AgglomerativeClustering with 'ward' only works with euclidean distances.",
        ):
            clusterer = NetworkClusterer(
                clusterer=None, n_clusters=2, linkage="ward"
            )

        with pytest.raises(
            ValueError,
            match="AgglomerativeClustering with 'ward' only works with euclidean distances.",
        ):
            clusterer = NetworkClusterer(
                clusterer="agglomerative", n_clusters=2, linkage="ward"
            )
        with pytest.raises(
            ValueError,
            match="AgglomerativeClustering with 'ward' only works with euclidean distances.",
        ):
            clusterer = NetworkClusterer(
                clusterer=AgglomerativeClustering(
                    n_clusters=2, metric="precomputed"
                )
            )

    def test_non_symmetric_distances(self):
        """Test with a non-symmetric pairwise distance matrix."""
        distances = np.array([[0, 1, 2], [0, 0, 1], [2, 1, 0]])

        clusterer = NetworkClusterer()

        with pytest.raises(
            ValueError, match="pairwise_distances must be symmetric"
        ):
            clusterer.fit(pairwise_distances=distances)

    def test_non_square_distances(self):
        """Test with a non-square pairwise distance matrix."""
        distances = np.array([[0, 1, 2], [1, 0, 1]])

        clusterer = NetworkClusterer()

        with pytest.raises(
            ValueError, match="pairwise_distances must be a square matrix"
        ):
            clusterer.fit(pairwise_distances=distances)

    def test_non_numpy_distances(self):
        """Test with a non-numpy pairwise distance matrix."""
        distances = [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ]  # List instead of numpy array

        clusterer = NetworkClusterer()

        with pytest.raises(
            ValueError, match="pairwise_distances must be a numpy array"
        ):
            clusterer.fit(pairwise_distances=distances)

    def test_flatten_distance_matrix(self):
        """Test flattening of the distance matrix."""
        distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        clusterer = NetworkClusterer()
        flattened = clusterer._flatten_distance_matrix(distances)

        assert flattened.shape == (3, 2)
        expected_flattened = np.array([[1, 2], [1, 1], [2, 1]])
        np.testing.assert_array_equal(flattened, expected_flattened)
