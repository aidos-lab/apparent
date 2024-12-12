"Query and interact with our US Physician Referral Network Datasette."

import pandas as pd
import urllib
import os
import numpy as np
from itertools import product
from typing import Union
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
from apparent.networks import (
    NetworkBuilder,
    NetworkComparator,
    NetworkClusterer,
    NetworkDescriber,
    NetworkEmbedder,
)


class Apparent:

    def __init__(
        self,
        base_url=None,
        build_method=None,
        cluster_method=None,
        features=None,
        embed_method=None,
    ):
        self.base_url = self._load_base_url(base_url)
        self.data = None
        self.kilt = None  # Initialize a KILT to handle pairwise distances?

        self.builder = NetworkBuilder(build_method)
        # self.clusterer = NetworkClusterer(cluster_method)
        # self.describer = NetworkDescriber(features)
        # self.plotter = NetworkPlotter()

        self.networks = {}
        self.distances = {}

    def pull(self, sql_query):

        if os.path.isfile(sql_query):
            sql_query = self._read_query(sql_query)
        self.data = self._fetcher(sql_query)
        self.data.sort_values(["hsa", "year"], inplace=True)

        self.network_ids = list(
            product(
                self.data["hsa"].unique().tolist(),
                self.data["year"].unique().tolist(),
            )
        )

    def download_interactions(self):
        if self.data is None:
            print(
                "No data available. Please fetch data first using `Apparent.pull()`."
            )
            return None

        network_queries = self._batch_interaction_queries()

        interaction_data = []
        for query in tqdm(
            network_queries,
            desc="Downloading Physician Interactions",
            unit="HSA",
        ):
            data = self._fetcher(query)
            interaction_data.append(data)

        self.physician_interactions = pd.concat(interaction_data)

    def build_networks(self):
        """
        Builds networks for each HSA (Health Service Area) in the fetched data.

        For each unique combination of `hsanum` and `year`, this method:
        - Filters the data,
        - Uses the selected `build_method` to create a network graph,
        - Stores the resulting graphs in a dictionary with keys as (hsanum, year) tuples.
        """

        if not hasattr(self, "physician_interactions"):
            self.download_interactions()

        self.networks = {}
        # Group by 'hsan' and 'year' to process each unique combination
        grouped_data = self.physician_interactions.groupby(["hsa", "year"])
        for (hsa, year), group in grouped_data:
            # Call the builder's `build` method for each HSA/year combination
            graph = self.builder.build(group, hsa=hsa, year=year)
            self.networks[(hsa, year)] = graph

        self.data["Networks"] = self.networks.values()

    def add_features(
        self,
        node_features=["degree_centrality"],
        edge_features=["forman_curvature"],
    ):
        """
        Add specified network features to the networks and update the data.

        Parameters
        ----------
        features : list or dict
            Features to compute. Can be a list of feature names or a dictionary
            specifying separate node and edge features:
                - If a list, assumes all features apply to nodes.
                - If a dictionary, expects keys 'node_features' and 'edge_features'.

        Returns
        -------
        updated_data : pd.DataFrame
            Dataframe with computed network features added.
        """
        if not hasattr(self, "networks") or not self.networks:
            print("No networks available. Please build networks first.")
            return None

        if node_features is None and edge_features is None:
            print(
                "No features specified. Please provide node or edge features."
            )
            return None

        # List to collect all rows with computed features for updating the main data
        feature_rows = []

        for (hsanum, year), graph in self.networks.items():

            # Initialize NetworkDescriber for the current graph
            describer = NetworkDescriber(graph)

            # Compute node and edge features
            describer.compute_features(
                node_features=node_features, edge_features=edge_features
            )

            # Update the `updated_networks` dictionary with the modified graph
            self.networks[(hsanum, year)] = describer.G

        self.data["Networks"] = self.networks.values()

    def compare(
        self,
        measure="forman_curvature",
        **kwargs,
    ) -> Union[float, np.array]:
        self.comparator = NetworkComparator(self.networks.values())
        # if we already have curvature features, lets get these and pass these!
        if measure not in self.distances:
            D = self.comparator.compare(measure=measure, **kwargs)
            self.distances.update({measure: D})
        # Generate a pairwise distance matrix for the networks

    def embed(self, measure="forman_curvature"):
        if measure not in self.distances:
            print(
                f"No pairwise distance computed yet for {measure}. Computing now..."
            )
            self.compare(measure=measure)

        D = self.distances[measure]
        self.embedder = NetworkEmbedder(pairwise_distances=D)
        self.embedding = self.embedder.embed()

    def cluster_networks(self, measure="forman_curvature", clusterer=None):
        if measure not in self.distances:
            print(
                f"No pairwise distance computed yet for {measure}. Computing now..."
            )
            self.compare(measure=measure)
            self.embed(measure=measure)

        assert hasattr(self, "embedding"), "Please embed the networks first."

        self.clusterer = NetworkClusterer(clusterer=clusterer)
        self.clusterer.fit(pairwise_distances=self.distances[measure])

    def plot_embedding(self):
        if not hasattr(self, "embedding"):
            self.embed()
        if not hasattr(self, "clusterer"):
            self.cluster_networks()

        color = self.clusterer.labels_ if hasattr(self, "clusterer") else None
        unique_clusters = np.unique(color) if color is not None else []

        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            c=color,
            cmap="Pastel2",
        )

        # Create legend manually by matching cluster labels to colors
        if color is not None:
            handles = []
            for i in unique_clusters:
                handles.append(
                    plt.scatter(
                        [],
                        [],
                        color=scatter.cmap(scatter.norm(i)),
                        label=f"Cluster {i}",
                    )
                )
            plt.legend(handles=handles, title="Clusters")

        plt.show()

    def _fetcher(self, sql_query) -> pd.DataFrame:
        df = pd.DataFrame()
        try:
            # Encode the SQL query
            encoded_query = urllib.parse.quote(sql_query)

            # Construct the full URL
            url = f"{self.base_url}?sql={encoded_query}"

            # Fetch data using pandas
            df = pd.read_csv(url)

            assert (
                "hsa" in df.columns
            ), "Please select 'hsa' as an attribute in your query as is needed for `Apparent` to identify networks."

            assert (
                "year" in df.columns
            ), "Please select 'year' as an attribute in your query as is needed for `Apparent` to identify networks."

        except Exception as e:
            print(f"An error occurred: {e}")

        return df

    def _read_query(self, file_path):
        with open(file_path, "r") as file:
            return file.read()

    def _load_base_url(self, base_url):
        if type(base_url) == str:
            return base_url
        else:
            load_dotenv()
            return os.getenv("APPARENT_URL")

    def _batch_interaction_queries(self):
        queries = []

        for hsa, year in self.network_ids:
            query = f"""
                SELECT * FROM local_physician_interactions
                WHERE hsa = {int(hsa)} AND year = {int(year)};
            """
            queries.append(query)

        return queries
