from scott import Comparator
from itertools import combinations
import numpy as np


class NetworkComparator:
    """
    A class to compare networks using curvature-based metrics and Scott package.

    Attributes
    ----------
    networks : list
        A list of NetworkX graphs to compare.
    """

    def __init__(self, networks):
        """
        Initialize the NetworkComparison instance.

        Parameters
        ----------
        networks : list
            List of NetworkX graphs to compare.
        """
        self.networks = networks

    def compare(
        self, measure="forman_curvature", metric="landscape", **kwargs
    ) -> np.ndarray:
        """
        Compare networks using Scott's Comparator and return a pairwise distance matrix.

        Parameters
        ----------
        measure : str, optional
            The curvature measure to use for comparison (default: "forman_curvature").

        metric : str, optional
            The comparison metric to use (default: "landscape").

        Returns
        -------
        D : np.ndarray
            A symmetric pairwise distance matrix.
        """
        # Initialize the Comparator
        C = Comparator(measure=measure)

        # Initialize a distance matrix
        n = len(self.networks)
        D = np.zeros((n, n))

        # Compute pairwise distances
        for (i, G1), (j, G2) in combinations(enumerate(self.networks), 2):
            distance = C.fit_transform(G1, G2, metric=metric, **kwargs)
            D[i, j] = distance
            D[j, i] = distance

        return D
