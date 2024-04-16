"""Script for building networks from regional hospital referral information."""

import config
import networkx as nx
import numpy as np
import os
import sys
import pandas as pd
import pickle
import swifter


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


if __name__ == "__main__":
    print("Reading in csv")
    in_file = (
        config.DATA_PATH + "network_panel_undirected_local_hsa_edges.csv.gz"
    )
    edges_df = pd.read_csv(in_file)

    print("Edge Dataframe read from csv")

    print("Now removing duplicates...")
    df = edges_df[["hsanum", "year"]].drop_duplicates()

    print("Building Networks...")
    df.assign(
        G=df.apply(
            lambda row: build_network(
                edges_df=edges_df, hsanum=row["hsanum"], year=row["year"]
            ),
            axis=1,
        )
    )

    print("Assigning additional graph features...")

    df = df.assign(
        nnodes=df.G.swifter.apply(nx.number_of_nodes),
        nedges=df.G.swifter.apply(nx.number_of_edges),
        density=df.G.swifter.apply(nx.density),
        degree_assortativity=df.G.swifter.apply(
            nx.degree_assortativity_coefficient
        ),
    )

    out_file = os.path.join(
        config.OUTPUT_PATH, "nx_networks_undirected_local_hsa.pkl"
    )
    # print("Pickling Full Dataframe...")

    # df.to_pickle(out_file)
    print("----------------------------------")
    print(f"File stats: Num_graphs={len(df)}")
    print("Pickling individual graphs...")

    graphs_path = os.path.join(
        "/Users/jeremy.wayland/Desktop/projects/apparent/outputs/", "graphs"
    )
    print(f"Graphs will be saved to {graphs_path}")
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)

    for i, G in enumerate(df.G.values):

        out_file = os.path.join(graphs_path, f"graph_{i}.pkl")
        result = {"graph": G}
        with open(out_file, "wb") as f:
            pickle.dump(result, f)

    sys.exit(0)
