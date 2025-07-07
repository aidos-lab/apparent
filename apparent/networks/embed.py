from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np


class NetworkEmbedder:
    """
    A class for embedding networks into a lower-dimensional space using pairwise distances or raw data.

    This class provides functionality to embed networks into a lower-dimensional space
    for visualization and analysis. It supports both precomputed distance matrices
    and raw data, using t-SNE as the default embedding method.

    Parameters
    ----------
    pairwise_distances : np.ndarray, optional
        A pairwise distance matrix (n x n) between networks. Used for precomputed
        embedding methods.
    method : object, optional
        An embedding method class. Default is TSNE from scikit-learn.
    **kwargs : dict
        Additional arguments passed to the embedding method.

    Attributes
    ----------
    D : np.ndarray or None
        A pairwise distance matrix (n x n) between networks.
    method : object
        An embedding method class (default: TSNE).
    scaler : StandardScaler
        Scaler for standardizing the embedding output.
    kwargs : dict
        Additional arguments for the embedding method.

    Examples
    --------
    >>> import numpy as np
    >>> from apparent.networks import NetworkEmbedder
    >>> 
    >>> # Create sample distance matrix
    >>> distances = np.random.rand(10, 10)
    >>> distances = (distances + distances.T) / 2  # Make symmetric
    >>> np.fill_diagonal(distances, 0)  # Zero diagonal
    >>> 
    >>> # Create embedding
    >>> embedder = NetworkEmbedder(pairwise_distances=distances)
    >>> embedding = embedder.embed()
    >>> print(f"Embedding shape: {embedding.shape}")
    """

    def __init__(self, pairwise_distances=None, method=TSNE, **kwargs):
        """
        Initialize the NetworkEmbedder.

        Parameters
        ----------
        pairwise_distances : np.ndarray, optional
            Pairwise distance matrix for precomputed embedding methods.
            Should be a symmetric matrix with zeros on the diagonal.
        method : object, optional
            The embedding method class to use. Default is TSNE from scikit-learn.
            Must implement fit_transform method.
        **kwargs : dict
            Additional arguments for the embedding method. Common parameters
            for t-SNE include 'n_components', 'perplexity', 'learning_rate'.

        Examples
        --------
        >>> embedder = NetworkEmbedder()  # Default t-SNE
        >>> embedder = NetworkEmbedder(method=TSNE, perplexity=30.0)
        >>> embedder = NetworkEmbedder(pairwise_distances=distances)
        """
        self.D = pairwise_distances
        self.method = method
        self.scaler = StandardScaler()
        self.kwargs = kwargs

    def embed(self, data=None):
        """
        Perform the embedding based on the selected method.

        This method creates a low-dimensional embedding of the networks using either
        precomputed pairwise distances or raw data. The embedding is standardized
        using StandardScaler before being returned.

        Parameters
        ----------
        data : np.ndarray, optional
            Raw input data for embedding (used if `pairwise_distances` is None).
            Should be a 2D array with shape (n_samples, n_features).

        Returns
        -------
        embedding : np.ndarray
            Low-dimensional embedding of the networks, typically 2D for visualization.
            The embedding is standardized to have zero mean and unit variance.

        Raises
        ------
        ValueError
            If neither pairwise distances nor data are provided.

        Examples
        --------
        >>> embedder = NetworkEmbedder(pairwise_distances=distances)
        >>> embedding = embedder.embed()
        >>> print(f"Embedding shape: {embedding.shape}")  # (n_networks, 2)
        >>> 
        >>> # Using raw data instead
        >>> embedder = NetworkEmbedder()
        >>> embedding = embedder.embed(data=feature_matrix)
        """
        # Validate inputs
        if self.D is None and data is None:
            raise ValueError(
                "Either pairwise distances or data must be provided."
            )

        # Initialize the embedding model
        if self.D is not None:
            perplexity = self.kwargs.get("perplexity", int(len(self.D) / 2))
            # Use pairwise distances if available
            model = self.method(
                metric="precomputed",
                perplexity=perplexity,
                init="random",
                **self.kwargs
            )
            embedding = model.fit_transform(self.D)
        else:
            # Use raw data if no distance matrix is provided
            perplexity = self.kwargs.get("perplexity", int(len(data) / 2))
            model = self.method(perplexity=perplexity, **self.kwargs)
            embedding = model.fit_transform(data)

        return self.scaler.fit_transform(embedding)
