from scott import Comparator
from itertools import combinations
import numpy as np


class NetworkComparator:
    """
    A class to compare networks using curvature-based metrics and Scott package.

    This class provides functionality to compute pairwise distances between networks
    using various curvature measures. It leverages the Scott package for robust
    network comparison through topological data analysis.

    Parameters
    ----------
    networks : list
        A list of NetworkX graphs to compare.

    Attributes
    ----------
    networks : list
        A list of NetworkX graphs to compare.

    Examples
    --------
    >>> import networkx as nx
    >>> from apparent.networks import NetworkComparator
    >>> 
    >>> # Create sample networks
    >>> G1 = nx.karate_club_graph()
    >>> G2 = nx.erdos_renyi_graph(34, 0.1)
    >>> networks = [G1, G2]
    >>> 
    >>> # Compare networks
    >>> comparator = NetworkComparator(networks)
    >>> distances = comparator.compare(measure="forman_curvature")
    >>> print(f"Distance matrix shape: {distances.shape}")
    """

    def __init__(self, networks):
        """
        Initialize the NetworkComparator instance.

        Parameters
        ----------
        networks : list
            List of NetworkX graphs to compare. Each graph should be a valid
            NetworkX graph object.

        Examples
        --------
        >>> G1 = nx.karate_club_graph()
        >>> G2 = nx.erdos_renyi_graph(34, 0.1)
        >>> comparator = NetworkComparator([G1, G2])
        """
        self.networks = networks

    def compare(
        self,
        measure="forman_curvature",
        metric="landscape",
        weight=None,
        alpha=0.0,
        prob_fn=None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compare networks using Scott's Comparator and return a pairwise distance matrix.

        This method computes pairwise distances between all networks using the specified
        curvature measure and comparison metric. The computation is performed for all
        unique pairs of networks, creating a symmetric distance matrix.

        Parameters
        ----------
        measure : str, optional
            The curvature measure to use for comparison. Default is "forman_curvature".
            Available measures depend on the Scott package implementation.
        metric : str, optional
            The comparison metric to use for computing distances between curvature
            distributions. Default is "landscape".
        weight : str or None, optional
            The edge attribute to use as weight when computing curvatures. If None,
            edges are treated as unweighted. Default is None.
        alpha : float, optional
            The alpha parameter for the Comparator, used in certain curvature measures.
            Default is 0.0.
        prob_fn : callable or None, optional
            A probability function for the Comparator. If None, uses default probability
            function from Scott package. Default is None.
        **kwargs : dict, optional
            Additional keyword arguments passed to the Scott Comparator.

        Returns
        -------
        D : np.ndarray
            A symmetric pairwise distance matrix of shape (n_networks, n_networks).
            D[i, j] represents the distance between networks i and j.

        Raises
        ------
        ValueError
            If no networks are provided for comparison.

        Examples
        --------
        >>> G1 = nx.karate_club_graph()
        >>> G2 = nx.erdos_renyi_graph(34, 0.1)
        >>> comparator = NetworkComparator([G1, G2])
        >>> distances = comparator.compare(measure="forman_curvature", metric="landscape")
        >>> print(f"Distance between networks: {distances[0, 1]}")
        """
        # Initialize the Comparator
        C = Comparator(
            measure=measure, weight=weight, alpha=alpha, prob_fn=prob_fn
        )

        # Initialize a distance matrix
        n = len(self.networks)
        D = np.zeros((n, n))

        if n == 0:
            raise ValueError("No networks to compare.")

        # Compute pairwise distances
        for (i, G1), (j, G2) in combinations(enumerate(self.networks), 2):
            distance = C.fit_transform(G1, G2, metric=metric, **kwargs)
            D[i, j] = distance
            D[j, i] = distance

        return D
