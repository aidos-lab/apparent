from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import numpy as np


class NetworkClusterer:
    """
    A class to cluster networks using a pairwise distance matrix or user-provided cluster labels.

    This class provides a flexible interface for clustering networks based on their
    pairwise distance matrices. It supports multiple clustering algorithms and can
    handle both precomputed distance matrices and manual cluster assignments.

    Parameters
    ----------
    clusterer : object or str, optional
        A clustering algorithm instance or string identifier. If None, uses
        AgglomerativeClustering with 2 clusters. If string, must be one of
        'kmeans', 'agglomerative', or 'dbscan'.
    **kwargs : dict
        Additional keyword arguments passed to the clustering algorithm.

    Attributes
    ----------
    model : object
        The clustering model (e.g., KMeans, AgglomerativeClustering, or DBSCAN).
    labels_ : np.ndarray
        Cluster labels for each graph.

    Examples
    --------
    >>> import numpy as np
    >>> from apparent.networks import NetworkClusterer
    >>> 
    >>> # Create sample distance matrix
    >>> distances = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
    >>> 
    >>> # Cluster with default agglomerative clustering
    >>> clusterer = NetworkClusterer()
    >>> labels = clusterer.fit(pairwise_distances=distances)
    >>> print(f"Cluster labels: {labels}")
    """

    def __init__(self, clusterer=None, **kwargs):
        """
        Initialize the NetworkClusterer.

        Parameters
        ----------
        clusterer : object or str, optional
            A clustering algorithm instance or string identifier. If None, uses
            AgglomerativeClustering with 2 clusters and 'average' linkage.
            If string, must be one of 'kmeans', 'agglomerative', or 'dbscan'.
        **kwargs : dict
            Additional keyword arguments to initialize the clustering algorithm.
            For AgglomerativeClustering, common parameters include 'n_clusters',
            'linkage', and 'metric'.

        Raises
        ------
        ValueError
            If 'ward' linkage is specified with precomputed distances, or if
            an unsupported clustering method string is provided.

        Examples
        --------
        >>> clusterer = NetworkClusterer()  # Default agglomerative
        >>> clusterer = NetworkClusterer('kmeans', n_clusters=3)
        >>> clusterer = NetworkClusterer('dbscan', eps=0.5)
        """
        linkage = kwargs.get("linkage", "average")
        if linkage == "ward":
            raise ValueError(
                "AgglomerativeClustering with 'ward' only works with euclidean distances."
            )
        if clusterer is None:
            self.model = AgglomerativeClustering(
                linkage=linkage, n_clusters=2, **kwargs
            )
        elif isinstance(clusterer, str):
            # Create clustering model by name
            if clusterer.lower() == "kmeans":
                self.model = KMeans(**kwargs)
            elif clusterer.lower() == "agglomerative":
                self.model = AgglomerativeClustering(linkage=linkage, **kwargs)
            elif clusterer.lower() == "dbscan":
                self.model = DBSCAN(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported clustering method '{clusterer}'. Choose from 'kmeans', 'agglomerative', or 'dbscan'."
                )
        else:
            # Assume user provided a clustering model instance
            if (
                hasattr(clusterer, "linkage")
                and clusterer.metric == "precomputed"
                and clusterer.linkage == "ward"
            ):
                raise ValueError(
                    "AgglomerativeClustering with 'ward' only works with euclidean distances."
                )

            self.model = clusterer

        self.labels_ = None

    def fit(self, pairwise_distances=None, manual_labels=None):
        """
        Fit the clustering model to the pairwise distance matrix, or accept manual labels.

        This method performs clustering on the provided distance matrix or assigns
        manual labels. For algorithms that support precomputed distances, the
        distance matrix is used directly. Otherwise, the matrix is flattened to
        create feature vectors.

        Parameters
        ----------
        pairwise_distances : np.ndarray, optional
            A symmetric pairwise distance matrix (n x n) between graphs.
            Required unless manual_labels are provided.
        manual_labels : np.ndarray, optional
            User-defined cluster labels for each graph. If provided, clustering
            is skipped and these labels are used directly.

        Returns
        -------
        labels_ : np.ndarray
            Cluster labels for each graph. Also stored in self.labels_.

        Raises
        ------
        ValueError
            If neither pairwise_distances nor manual_labels are provided, or if
            the distance matrix is not square and symmetric.

        Examples
        --------
        >>> distances = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
        >>> clusterer = NetworkClusterer(n_clusters=2)
        >>> labels = clusterer.fit(pairwise_distances=distances)
        >>> print(f"Cluster labels: {labels}")
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

        Some clustering algorithms (like KMeans) require feature vectors rather than
        distance matrices. This method converts an (n x n) distance matrix into an
        (n x n-1) feature array by removing diagonal elements and flattening each row.

        Parameters
        ----------
        distance_matrix : np.ndarray
            A symmetric (n x n) distance matrix.

        Returns
        -------
        flattened_matrix : np.ndarray
            A feature array of shape (n, n-1) suitable for clustering algorithms
            that don't support precomputed distances.

        Examples
        --------
        >>> distances = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
        >>> clusterer = NetworkClusterer()
        >>> flattened = clusterer._flatten_distance_matrix(distances)
        >>> print(f"Flattened shape: {flattened.shape}")  # (3, 2)
        """
        n = distance_matrix.shape[0]
        return distance_matrix[np.arange(n)[:, None] != np.arange(n)].reshape(
            n, n - 1
        )
