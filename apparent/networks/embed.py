from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np


class NetworkEmbedder:
    """
    A class for embedding networks into a lower-dimensional space using pairwise distances or raw data.

    Attributes
    ----------
    D : np.ndarray, optional
        A pairwise distance matrix (n x n) between networks.
    method : object, optional
        An embedding method (default: TSNE).
    kwargs : dict
        Additional arguments for the embedding method.
    """

    def __init__(self, pairwise_distances=None, method=TSNE, **kwargs):
        """
        Initialize the NetworkEmbedder.

        Parameters
        ----------
        pairwise_distances : np.ndarray, optional
            Pairwise distance matrix for precomputed embedding methods.

        method : object, optional
            The embedding method to use (default: TSNE).

        **kwargs
            Additional arguments for the embedding method.
        """
        self.D = pairwise_distances
        self.method = method
        self.scaler = StandardScaler()
        self.kwargs = kwargs

    def embed(self, data=None):
        """
        Perform the embedding based on the selected method.

        Parameters
        ----------
        data : np.ndarray, optional
            Raw input data for embedding (used if `pairwise_distances` is None).

        Returns
        -------
        embedding : np.ndarray
            Low-dimensional embedding of the networks.
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
