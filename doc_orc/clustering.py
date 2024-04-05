"Clustering Physician Referral Networks."

import pandas as pd
import os
import argparse
import sys
import numpy as np
import itertools
from scipy.sparse import coo_array
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

from topology import calculate_persistence_diagrams
from utils import make_node_filtration


def plot_dendrogram(model, ids):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(
        linkage_matrix,
        labels=ids,
    )
    # Plot dendrogram
    plt.title("2014 Sample Physician Referral Networks")
    plt.xlabel("hsanum")
    plt.ylabel("Curvature Filtrations Distance")
    plt.show()


def fit_landscapes(
    data: pd.DataFrame, id_col: str = "hsanum", curvature_col: str = "OR_0.0"
):

    graphs = data["G"]
    curvatures = data[curvature_col]

    iterator = dict(
        zip(
            data[id_col],
            zip(graphs, curvatures),
        )
    )
    landscapes = {}
    for id_, (G, curvature) in iterator.items():
        if curvature is None:
            continue
        G = make_node_filtration(G, curvature, attribute_name="curvature")
        dgm = calculate_persistence_diagrams(G, "curvature", "curvature")
        landscapes[id_] = {i: D.fit_landscape() for i, D in enumerate(dgm)}
    return landscapes


def pairwise_landscape_distances(L1, L2):
    """Compute pairwise distances between landscapes."""
    diff = dict()
    for i in L1.keys():
        diff[i] = L1[i] - L2[i]
    norms = {k: np.linalg.norm(v) for k, v in diff.items()}
    return sum(norms.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file",
    )
    parser.add_argument(
        "--curvature",
        type=str,
        default="OR_0.0",
        help="Column Label for Curvature Filtration",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    # Load the data
    assert os.path.isfile(args.input), f"File not found: {args.input}"
    data = pd.read_pickle(args.input)

    # Fit the landscapes
    descriptors = fit_landscapes(data, curvature_col=args.curvature)

    ids = list(descriptors.keys())
    landscapes = list(descriptors.values())
    N = len(landscapes)
    # Compute pairwise distances
    pairs = list(itertools.combinations(range(N), 2))

    distances = list()
    for i, j in pairs:
        L1, L2 = landscapes[i], landscapes[j]
        d = pairwise_landscape_distances(L1, L2)
        distances.append((d, i, j))

    values = list(map(lambda tup: tup[0], distances))
    rows = list(map(lambda tup: tup[1], distances))
    cols = list(map(lambda tup: tup[2], distances))

    M = coo_array(
        (values, (rows, cols)),
        shape=(len(landscapes), len(landscapes)),
    ).todense()
    M += M.T

    clustering_model = AgglomerativeClustering(
        metric="precomputed",
        linkage="complete",
        compute_distances=True,
        distance_threshold=None,
        n_clusters=3,
    )
    clustering_model.fit(M)

    plot_dendrogram(clustering_model, ids)
