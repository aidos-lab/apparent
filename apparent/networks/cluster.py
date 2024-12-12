from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import numpy as np


class NetworkClusterer:
    """
    A class to cluster networks using a pairwise distance matrix or user-provided cluster labels.

    Attributes
    ----------
    model : object
        The clustering model (e.g., KMeans, AgglomerativeClustering, or DBSCAN).
    labels_ : np.ndarray
        Cluster labels for each graph.
    """

    def __init__(self, clusterer=None, **kwargs):
        """
        Initialize the NetworkClusterer.

        Parameters
        ----------
        clusterer : object, optional
            A clustering algorithm instance (default: AgglomerativeClustering with 2 clusters).

        **kwargs
            Additional keyword arguments to initialize the default clusterer.
        """
        if clusterer is None:
            self.model = AgglomerativeClustering(n_clusters=2, **kwargs)
        elif isinstance(clusterer, str):
            # Create clustering model by name
            if clusterer.lower() == "kmeans":
                self.model = KMeans(**kwargs)
            elif clusterer.lower() == "agglomerative":
                self.model = AgglomerativeClustering(**kwargs)
            elif clusterer.lower() == "dbscan":
                self.model = DBSCAN(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported clustering method '{clusterer}'. Choose from 'kmeans', 'agglomerative', or 'dbscan'."
                )
        else:
            # Assume user provided a clustering model instance
            self.model = clusterer

        self.labels_ = None

    def fit(self, pairwise_distances=None, manual_labels=None):
        """
        Fit the clustering model to the pairwise distance matrix, or accept manual labels.

        Parameters
        ----------
        pairwise_distances : np.ndarray, optional
            A symmetric pairwise distance matrix (n x n) between graphs.

        manual_labels : np.ndarray, optional
            User-defined cluster labels for each graph.

        Returns
        -------
        labels_ : np.ndarray
            Cluster labels for each graph.
        """
        # If user provides manual labels, skip clustering and return these labels
        if manual_labels is not None:
            if not isinstance(manual_labels, np.ndarray):
                raise ValueError("manual_labels must be a numpy array.")
            if manual_labels.ndim != 1:
                raise ValueError("manual_labels must be a 1D array.")
            self.labels_ = manual_labels
            return self.labels_

        # Check if a pairwise distance matrix is provided
        if pairwise_distances is None:
            raise ValueError(
                "pairwise_distances must be provided unless manual_labels are used."
            )

        # Ensure the distance matrix is a numpy array
        if not isinstance(pairwise_distances, np.ndarray):
            raise ValueError("pairwise_distances must be a numpy array.")

        # Ensure the distance matrix is a square symmetric matrix
        if pairwise_distances.shape[0] != pairwise_distances.shape[1]:
            raise ValueError("pairwise_distances must be a square matrix.")
        if not np.allclose(pairwise_distances, pairwise_distances.T, atol=1e-8):
            raise ValueError("pairwise_distances must be symmetric.")

        # Handle clustering model
        if hasattr(self.model, "metric") and self.model.metric == "precomputed":
            self.labels_ = self.model.fit_predict(pairwise_distances)
        else:
            # Flatten distance matrix if the clustering method does not support precomputed distances
            flattened_distances = pairwise_distances
            if (
                flattened_distances.ndim == 2
                and flattened_distances.shape[0] == flattened_distances.shape[1]
            ):
                flattened_distances = self._flatten_distance_matrix(
                    pairwise_distances
                )
            self.labels_ = self.model.fit_predict(flattened_distances)

        return self.labels_

    def _flatten_distance_matrix(self, distance_matrix):
        """
        Flatten a pairwise distance matrix to a 2D array.

        Some clustering algorithms (like KMeans) require feature vectors rather than distance matrices.
        This method converts an (n x n) distance matrix into an (n x n-1) feature array by removing diagonal
        elements and flattening each row.

        Parameters
        ----------
        distance_matrix : np.ndarray
            A symmetric (n x n) distance matrix.

        Returns
        -------
        flattened_matrix : np.ndarray
            A feature array suitable for clustering algorithms that don't support precomputed distances.
        """
        n = distance_matrix.shape[0]
        return distance_matrix[np.arange(n)[:, None] != np.arange(n)].reshape(
            n, n - 1
        )
