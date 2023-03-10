"""Visualise histogram of filtration values."""

import argparse
import warnings

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances


def laplacian_eigenvalues(G):
    """Calculate Laplacian and return eigenvalues."""
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        return nx.laplacian_spectrum(G)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("FILE", type=str, help="Input file (in `.g6` format)")

    args = parser.parse_args()
    graphs = nx.read_graph6(args.FILE)

    n = len(graphs)
    n_rows = int(np.ceil(np.sqrt(n)))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_rows, squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    X = []

    for graph, ax in zip(graphs, axes.ravel()):
        ax.set_visible(True)

        values = laplacian_eigenvalues(graph)
        sns.histplot(values, ax=ax)

        X.append(values)

    X = np.asarray(X)
    print(np.max(pairwise_distances(X)))

    plt.tight_layout(pad=0.5)
    plt.show()
