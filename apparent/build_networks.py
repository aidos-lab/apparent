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

    print(G)
    return G


if __name__ == "__main__":
    print("Reading in csv")
    in_file = (
        config.DATA_PATH + "network_panel_undirected_local_hsa_edges.csv.gz"
    )

    edges_df = pd.read_csv(in_file)
    print("Edge Dataframe read from csv")

    print("Now removing duplicates...")
    df = edges_df[["hsanum", "year"]].drop_duplicates(keep="first")

    print("Building Networks...")
    df = df.assign(
        graph=df.apply(
            lambda row: build_network(
                edges_df=edges_df, hsanum=row["hsanum"], year=row["year"]
            ),
            axis=1,
        )
    )
    print("Assigning additional graph features...")

    # TODO: Remove and add to compute features script
    # ALso would like a separate script to combine geographical/medicare data
    df = df.assign(
        nnodes=df["graph"].swifter.apply(nx.number_of_nodes),
        nedges=df["graph"].swifter.apply(nx.number_of_edges),
        density=df["graph"].swifter.apply(nx.density),
        degree_assortativity=df["graph"].swifter.apply(
            nx.degree_assortativity_coefficient
        ),
    )

    compact_df = df.drop(columns=["graph"])
    out_file = os.path.join(config.OUTPUT_PATH, "networks_data.pkl")
    print(f"Saving networks to {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(compact_df, f)
    print("----------------------------------")
    print(f"File stats: Num_graphs={len(df)}")
    print("Pickling individual graphs...")

    graphs_path = config.OUTPUT_PATH + "graphs/"
    print(f"Graphs will be saved to {graphs_path}")
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)

    relevant_fields = [
        "graph",
        "nnodes",
        "nedges",
        "density",
        "degree_assortativity",
        "year",
        "hsanum",
    ]

    for i, row in df[relevant_fields].iterrows():
        data = {col: row[col] for col in relevant_fields}
        out_file = os.path.join(graphs_path, f"graph_{i}.pkl")
        print(f"Pickling graph {i} to {out_file}")
        if os.path.exists(out_file):
            print("Path Exists!")
            with open(out_file, "rb") as f:
                saved_network = pickle.load(f)
                assert nx.is_isomorphic(row["graph"], saved_network["graph"])
            data.pop("graph")
            data = {**saved_network, **data}
            print(data)
        with open(out_file, "wb") as f:
            pickle.dump(data, f)
    print("Finished pickling individual graphs!")
    sys.exit(0)
