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


class NetworkClusterer:
    pass


# TODO: Suppport different clustering algorithms?
def fit_landscapes(
    data: dict,
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
