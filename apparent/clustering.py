"Clustering Physician Referral Networks."

import pandas as pd
import os
import argparse
import sys
import numpy as np
import itertools
from scipy.sparse import coo_array
import random
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

import config
from topology import calculate_persistence_diagrams
from utils import (
    make_node_filtration,
    load_graphs,
    plot_phate_embedding,
)


def fit_landscapes(
    data: dict,
    id_col: str = "hsanum",
    filtration: str = "OR_0",
):
    landscapes = {}
    for network_id in data:
        G = data[network_id]["graph"]
        curvature = data[network_id][filtration]
        if curvature is None:
            continue
        G = make_node_filtration(G, curvature, attribute_name="curvature")

        dgm = calculate_persistence_diagrams(G, "curvature", "curvature")
        # TODO: Treat trivial diagrams better
        try:
            landscapes[network_id] = {
                i: D.fit_landscape() for i, D in enumerate(dgm)
            }
        except Exception as e:
            print(f"Error fitting landscape for {network_id}: {e}")
            print(dgm)
            print()
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
        "--graphs_dir",
        type=str,
        help="Directory pointing to precomputed graphs.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2014,
        help="Year of interest. Default is 2014.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="OR_0",
        help="Column Label for Graph Feature of interest.",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    # Load the data
    assert os.path.isdir(args.graphs_dir), f"File not found: {args.graphs_dir}"

    network_data = load_graphs(
        args.graphs_dir, feature=args.feature, year=args.year
    )

    print(f"Loaded {len(network_data)} networks.")
    # Fit the landscapes
    descriptors = fit_landscapes(network_data, filtration=args.feature)

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

    # Plot PHATE Embedding
    fig, X = plot_phate_embedding(
        M,
        n_components=2,
        knn=10,
        decay=40,
        njobs=4,
        year=args.year,
        feature=args.feature,
    )

    plt.show()
    figure_out = os.path.join(
        config.OUTPUT_PATH,
        f"figures/phate_embedding_{args.year}_{args.feature}.png",
    )
    plt.savefig(figure_out)

    # TODO: Implement Set Cover Algorithm
    # TODO: Link prediction based on OR feature distribution (sampling)
    # TODO: Link prediction based on maximal move towards most similar "affluent representative"
