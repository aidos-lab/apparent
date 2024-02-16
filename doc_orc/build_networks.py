"""Script for building networks from regional hospital referal information."""


import config  # edit data path in here
import networkx as nx
import numpy as np
import pandas as pd
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


print("Reading in csv")
edges_df = pd.read_csv(
    config.DATA_PATH + "/network_panel_undirected_local_hsa_edges.csv.gz"
)

print("Edge Dataframe read from csv")

print("Now removing duplicates...")
df = edges_df[["hsanum", "year"]].drop_duplicates()


print("Building Networks...")
df = df.assign(
    G=df.swifter.apply(
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
    degree_assortativity=df.G.swifter.apply(nx.degree_assortativity_coefficient),
)





print(df.head())
print("Writing Pickle...")

df.to_pickle(config.DATA_PATH + "/nx_networks_undirected_local_hsa.pkl")

print("Finished pickling")
print("----------------------------------")
print(f"File stats: Num_graphs={len(df)}, Columns={df.columns}")
