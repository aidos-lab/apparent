"Query and interact with our US Physician Referral Network Datasette."

import pandas as pd
import urllib
import os
from typing import Union
from dotenv import load_dotenv
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
        base_url,
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

    def fetch_data(self, sql_query):
        try:
            # Encode the SQL query
            encoded_query = urllib.parse.quote(sql_query)

            # Construct the full URL
            url = f"{self.base_url}?sql={encoded_query}"

            # Fetch data using pandas
            self.data = pd.read_csv(url)

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def build_networks(self):
        """Build networks for HSAs in the users query."""
        # TODO: How can we allow more advanced users to build their own networks?
        # Can probably use swifter to add a GRAPH column to existing dataframe.
        pass

    def add_features(self, features):
        """Add network features to fetched data."""
        pass

    def compare(
        self, HSAs, measure="forman_curvature", **kwargs
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

    def _read_query(self, file_path):
        with open(file_path, "r") as file:
            return file.read()

    def _load_base_url(self, base_url):
        if type(base_url) == str:
            return base_url
        else:
            load_dotenv()
            return os.getenv("APPARENT_URL")

    def _save_networks(self):
        # TODO: save indidivual networks to disk? einfach in a dataframe?
        pass
