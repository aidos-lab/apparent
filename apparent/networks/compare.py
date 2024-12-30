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

        Parameters
        ----------
        measure : str, optional
            The curvature measure to use for comparison (default: "forman_curvature").

        metric : str, optional
            The comparison metric to use (default: "landscape").

        weight : str or None, optional
            The edge attribute to use as weight (default: None).

        alpha : float, optional
            The alpha parameter for the Comparator (default: 0.0).

        prob_fn : callable or None, optional
            A probability function for the Comparator (default: None).

        **kwargs : dict, optional
            Additional keyword arguments for the Comparator.

        Returns
        -------
        D : np.ndarray
            A symmetric pairwise distance matrix.
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
