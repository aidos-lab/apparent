"Query and interact with our US Physician Referral Network Datasette."

import pandas as pd
import urllib
import os
import numpy as np
from itertools import product
from typing import Union
from dotenv import load_dotenv
from tqdm import tqdm
from apparent.networks import (
    NetworkBuilder,
    NetworkClusterer,
    NetworkDescriber,
    NetworkEmbedder,
    NetworkPlotter,
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
        self.clusterer = NetworkClusterer(cluster_method)
        self.describer = NetworkDescriber(features)
        self.embedder = NetworkEmbedder(embed_method)
        self.plotter = NetworkPlotter()

    def pull(self, sql_query):
        self.data = self._fetcher(sql_query)
        self.hsas = self.data["hsa"].unique().tolist()
        self.years = self.data["year"].unique().tolist()

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

        # Dictionary to store the resulting networks
        self.networks = {}

        # Group by 'hsan' and 'year' to process each unique combination
        grouped_data = self.physician_interactions.groupby(["hsa", "year"])

        for (hsa, year), group in grouped_data:
            # Call the builder's `build` method for each HSA/year combination
            graph = self.builder.build(group, hsa=hsa, year=year)

            # Store the resulting graph in the dictionary
            self.networks[(hsa, year)] = graph

    def add_features(self, features: Union[list, dict]):
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

        if isinstance(features, list):
            # Assume all features are node features if a simple list is provided
            node_features = features
            edge_features = []
        elif isinstance(features, dict):
            # If a dictionary is provided, use it to separate node and edge features
            node_features = features.get("node_features", [])
            edge_features = features.get("edge_features", [])
        else:
            raise ValueError("Features should be a list or a dictionary.")

        # List to collect all rows with computed features for updating the main data
        feature_rows = []

        # Rebuild `self.networks` with graphs containing the features
        updated_networks = {}

        for (hsanum, year), graph in self.networks.items():
            print(f"Adding features for network (HSA: {hsanum}, Year: {year})")

            # Initialize NetworkDescriber for the current graph
            describer = NetworkDescriber(graph)

            # Compute node and edge features
            describer.compute_features(
                node_features=node_features, edge_features=edge_features
            )

            # Add features to the graph as attributes
            features_dict = describer.get_features()

            # Update node features
            for feature, values in features_dict.items():
                if feature in node_features:
                    nx.set_node_attributes(graph, values, name=feature)
                elif feature in edge_features:
                    nx.set_edge_attributes(graph, values, name=feature)

            # Update the `updated_networks` dictionary with the modified graph
            updated_networks[(hsanum, year)] = graph

            # Extract node features and add HSA/year info
            for node, data in graph.nodes(data=True):
                feature_row = {"hsanum": hsanum, "year": year, "node": node}
                feature_row.update(data)
                feature_rows.append(feature_row)

        # Reassign `self.networks` to contain graphs with features
        self.networks = updated_networks

        # Create a DataFrame from the collected feature rows
        feature_data = pd.DataFrame(feature_rows)

        # Merge the features back into the original data, if possible
        if "node" in self.data.columns:
            self.data = pd.merge(
                self.data,
                feature_data,
                on=["hsanum", "year", "node"],
                how="left",
            )
        else:
            self.data = self.data.merge(
                feature_data, on=["hsanum", "year"], how="left"
            )

        print("Features successfully added to the networks and data.")
        return self.data

    def compare(
        self, HSA1, HSA2, measure="forman_curvature", **kwargs
    ) -> Union[float, np.array]:
        # READ networks from self.data
        # Fit Filtrations with KILT and return float
        pass

    def embed_networks(self):
        # TODO: Pairwise distances between networks using KILT
        # TODO: Embed networks via TSNE or other method
        pass

    def cluster_networks(self):
        pass

    def plot_network(self, network_ids):
        # TODO: plot graph(s)
        pass

    def plot_feature_distribution(self, feature):
        # TODO: Distribution
        pass

    def plot_embedding(self):
        pass

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

        for hsa, year in product(self.hsas, self.years):
            query = f"""
                SELECT * FROM local_physician_interactions
                WHERE hsa = {int(hsa)} AND year = {int(year)};
            """
            queries.append(query)

        return queries

    def _save_networks(self):
        # TODO: save indidivual networks to disk? einfach in a dataframe?
        pass
