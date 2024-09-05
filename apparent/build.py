"""Script for building networks from regional hospital referral information."""

import networkx as nx
import numpy as np
import os
import sys
import pandas as pd
import pickle
import swifter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


import apparent.config as config
from apparent.curvature import forman_curvature
from apparent.utils import convert_np_array


def build_network(edges_df, hsanum, year):
    """Create a networkx graph given a dataframe of edges, a region, and year."""

    # initialize an undirected graph G
    G = nx.Graph()

    # populate G
    G = nx.from_pandas_edgelist(
        df=edges_df[(edges_df.hsanum == hsanum) & (edges_df.year == year)],
        source="npi_a",
        target="npi_b",
        edge_attr=["a2b", "b2a"],
    )

    # sanity check
    assert G.is_directed() is False
    assert G.is_multigraph() is False

    return G


def process_row(row):
    A, nodes, curvature = build_network(edges_df, row["hsanum"], row["year"])
    return pd.Series({"adjacency": A, "nodes": nodes, "curvature": curvature})


if __name__ == "__main__":
    print("Reading in csv")
    in_file = (
        config.DATA_PATH + "network_panel_undirected_local_hsa_edges.csv.gz"
    )

    edges_df = pd.read_csv(in_file)
    print("Edge Dataframe read from csv")

    print("Now removing duplicates...")
    df = edges_df[["hsanum", "year"]].drop_duplicates(keep="first")

    df = df.assign(
        graph=df.apply(
            lambda row: build_network(
                edges_df=edges_df, hsanum=row["hsanum"], year=row["year"]
            ),
            axis=1,
        ),
    )

    outPath = "/Users/jeremy.wayland/Desktop/dev/apparent/outputs/all_graphs/"

    for index, row in tqdm(df.iterrows(), desc="Rows"):
        data = {
            "hsa": row["hsanum"],
            "year": row["year"],
            "graph": row["graph"],
        }
        outFile = os.path.join(outPath, f"graph_{index}.pkl")
        with open(outFile, "wb") as f:
            pickle.dump(data, f)
