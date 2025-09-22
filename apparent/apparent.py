"""
Apparent: A Comprehensive Interface for US Physician Referral Network Analysis

The Apparent class provides a user-friendly interface to query, build, analyze,
and visualize physician referral networks from the US healthcare system. It integrates
various functionalities including data fetching, network construction, feature
computation, network comparison, clustering, and embedding.
"""
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
    """
    Query and interact with our US Physician Referral Network Datasette.

    The Apparent class provides a comprehensive interface for analyzing physician referral 
    networks from the US healthcare system. It integrates network building, feature 
    computation, comparison, clustering, and visualization capabilities.

    Parameters
    ----------
    base_url : str, optional
        The base URL for the Datasette instance. If not provided, will attempt to 
        load from the APPARENT_URL environment variable.

    Attributes
    ----------
    base_url : str
        The base URL for the Datasette instance.
    data : pd.DataFrame or None
        The fetched data from the Datasette.
    kilt : object or None
        KILT object for handling pairwise distances (future use).
    networks : dict
        Dictionary storing network graphs with (hsa, year) as keys.
    distances : dict
        Dictionary storing pairwise distance matrices for different measures.
    network_ids : list
        List of (hsa, year) tuples identifying unique networks.
    physician_interactions : pd.DataFrame
        DataFrame containing physician interaction data.
    builder : NetworkBuilder
        Instance for building network graphs.
    comparator : NetworkComparator
        Instance for comparing networks.
    embedder : NetworkEmbedder
        Instance for embedding networks.
    clusterer : NetworkClusterer
        Instance for clustering networks.
    embedding : np.ndarray
        Low-dimensional embedding of networks.

    Examples
    --------
    >>> import apparent
    >>> app = apparent.Apparent()
    >>> 
    >>> # Fetch data from a SQL query
    >>> query = "SELECT * FROM physician_data WHERE year >= 2020"
    >>> app.pull(query)
    >>> 
    >>> # Build networks for each HSA and year
    >>> app.build_networks()
    >>> 
    >>> # Add network features
    >>> app.add_features(node_features=["degree", "clustering"], 
    ...                  edge_features=["forman_curvature"])
    >>> 
    >>> # Compare networks and create embedding
    >>> app.compare(measure="forman_curvature")
    >>> app.embed()
    >>> 
    >>> # Cluster networks
    >>> app.cluster_networks()
    >>> 
    >>> # Visualize the embedding
    >>> app.plot_embedding()
    """

    def __init__(
        self,
        base_url=None,
    ):
        self.base_url = self._load_base_url(base_url)
        self.data = None
        self.kilt = None  # Initialize a KILT to handle pairwise distances?

        self.networks = {}
        self.distances = {}

    def pull(self, sql_query):
        """
        Execute a SQL query against the Datasette and store the results.

        Parameters
        ----------
        sql_query : str
            SQL query string to execute, or path to a file containing the query.
            The query must include 'hsa' and 'year' columns for network identification.

        Returns
        -------
        None
            Results are stored in self.data and self.network_ids attributes.

        Notes
        -----
        The query must return data with 'hsa' and 'year' columns, which are used
        to identify unique networks. The data is sorted by 'hsa' and 'year' after
        fetching.

        Examples
        --------
        >>> app = Apparent()
        >>> app.pull("SELECT * FROM physician_data WHERE year >= 2020")
        >>> print(app.data.head())
        """

        if os.path.isfile(sql_query):
            sql_query = self._read_query(sql_query)
        self.data = self._fetcher(sql_query)
        
        # Validate required columns exist
        required_columns = ["hsa", "year"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Query results must contain 'hsa' and 'year' columns. Missing: {missing_columns}")
        
        self.data.sort_values(["hsa", "year"], inplace=True)

        self.network_ids = list(
            product(
                self.data["hsa"].unique().tolist(),
                self.data["year"].unique().tolist(),
            )
        )

    def download_interactions(self):
        """
        Download physician interaction data for all network identifiers.

        This method fetches detailed interaction data for each unique (hsa, year)
        combination identified in the pulled data. The interactions are downloaded
        in batches with a progress bar.

        Returns
        -------
        None
            Results are stored in self.physician_interactions attribute.

        Raises
        ------
        ValueError
            If no data is available (must call pull() first).

        Examples
        --------
        >>> app = Apparent()
        >>> app.pull("SELECT * FROM physician_data WHERE year >= 2020")
        >>> app.download_interactions()
        >>> print(f"Downloaded {len(app.physician_interactions)} interactions")
        """
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

    def build_networks(self, build_method=None):
        """
        Build networks for each HSA (Health Service Area) in the fetched data.

        For each unique combination of `hsa` and `year`, this method:
        - Filters the physician interaction data
        - Uses the selected `build_method` to create a network graph
        - Stores the resulting graphs in a dictionary with keys as (hsa, year) tuples

        Parameters
        ----------
        build_method : callable, optional
            A custom function that accepts the dataframe and additional keyword 
            arguments and returns a NetworkX graph. If not provided, the default 
            standard_build method is used.

        Returns
        -------
        None
            Results are stored in self.networks attribute and self.data is updated
            with a 'Networks' column.

        Notes
        -----
        If physician interactions haven't been downloaded yet, this method will
        automatically call download_interactions() first.

        Examples
        --------
        >>> app = Apparent()
        >>> app.pull("SELECT * FROM physician_data WHERE year >= 2020")
        >>> app.build_networks()
        >>> print(f"Built {len(app.networks)} networks")
        """
        self.builder = NetworkBuilder(build_method)

        if not hasattr(self, "physician_interactions") or self.physician_interactions is None:
            self.download_interactions()

        self.networks = {}
        # Group by 'hsa' and 'year' to process each unique combination
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

        This method computes node and edge features for each network using the
        NetworkDescriber class and updates the network graphs with these features
        as attributes.

        Parameters
        ----------
        node_features : list, optional
            List of node features to compute. Available options include:
            'degree', 'clustering', 'betweenness', 'closeness', 'pagerank'.
            Default is ['degree_centrality'].
        edge_features : list, optional
            List of edge features to compute. Available options include:
            'edge_betweenness' and any curvature measure from scott.kilt.
            Default is ['forman_curvature'].

        Returns
        -------
        None
            Networks are updated in place and self.data is updated with a
            'Networks' column containing the enhanced graphs.

        Raises
        ------
        ValueError
            If no networks are available (must call build_networks() first).

        Examples
        --------
        >>> app = Apparent()
        >>> app.pull("SELECT * FROM physician_data WHERE year >= 2020")
        >>> app.build_networks()
        >>> app.add_features(node_features=["degree", "clustering"],
        ...                  edge_features=["forman_curvature", "edge_betweenness"])
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
        """
        Compare networks using curvature-based metrics.

        This method computes pairwise distances between all networks using the
        specified measure and stores the resulting distance matrix.

        Parameters
        ----------
        measure : str, optional
            The curvature measure to use for comparison. Default is 'forman_curvature'.
            Must be a valid curvature measure from scott.kilt.
        **kwargs : dict
            Additional keyword arguments passed to the NetworkComparator.

        Returns
        -------
        np.ndarray
            Symmetric pairwise distance matrix of shape (n_networks, n_networks).
            Also stored in self.distances[measure].

        Examples
        --------
        >>> app = Apparent()
        >>> app.pull("SELECT * FROM physician_data WHERE year >= 2020")
        >>> app.build_networks()
        >>> app.add_features(edge_features=["forman_curvature"])
        >>> distances = app.compare(measure="forman_curvature")
        >>> print(f"Distance matrix shape: {distances.shape}")
        """
        self.comparator = NetworkComparator(self.networks.values())
        # if we already have curvature features, lets get these and pass these!
        if measure not in self.distances:
            D = self.comparator.compare(measure=measure, **kwargs)
            self.distances.update({measure: D})
        # Generate a pairwise distance matrix for the networks

    def embed(self, measure="forman_curvature"):
        """
        Embed networks into a lower-dimensional space using t-SNE.

        This method creates a 2D embedding of the networks based on their pairwise
        distances using the specified measure.

        Parameters
        ----------
        measure : str, optional
            The curvature measure to use for embedding. Default is 'forman_curvature'.
            Must match a measure used in compare().

        Returns
        -------
        None
            Results are stored in self.embedding attribute.

        Notes
        -----
        If pairwise distances for the specified measure haven't been computed yet,
        this method will automatically call compare() first.

        Examples
        --------
        >>> app = Apparent()
        >>> app.pull("SELECT * FROM physician_data WHERE year >= 2020")
        >>> app.build_networks()
        >>> app.add_features(edge_features=["forman_curvature"])
        >>> app.embed(measure="forman_curvature")
        >>> print(f"Embedding shape: {app.embedding.shape}")
        """
        if measure not in self.distances:
            print(
                f"No pairwise distance computed yet for {measure}. Computing now..."
            )
            self.compare(measure=measure)

        D = self.distances[measure]
        self.embedder = NetworkEmbedder(pairwise_distances=D)
        self.embedding = self.embedder.embed()

    def cluster_networks(self, measure="forman_curvature", clusterer=None):
        """
        Cluster networks based on their pairwise distances.

        This method applies clustering to the networks using their pairwise distance
        matrix. If an embedding hasn't been computed yet, it will be created first.

        Parameters
        ----------
        measure : str, optional
            The curvature measure to use for clustering. Default is 'forman_curvature'.
            Must match a measure used in compare().
        clusterer : object, optional
            A clustering algorithm instance. If None, uses AgglomerativeClustering
            with 2 clusters.

        Returns
        -------
        None
            Results are stored in self.clusterer attribute, with cluster labels
            available at self.clusterer.labels_.

        Notes
        -----
        If pairwise distances or embedding for the specified measure haven't been
        computed yet, this method will automatically call compare() and embed() first.

        Examples
        --------
        >>> app = Apparent()
        >>> app.pull("SELECT * FROM physician_data WHERE year >= 2020")
        >>> app.build_networks()
        >>> app.add_features(edge_features=["forman_curvature"])
        >>> app.cluster_networks(measure="forman_curvature")
        >>> print(f"Cluster labels: {app.clusterer.labels_}")
        """
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
        """
        Plot the 2D embedding of networks with cluster colors.

        This method creates a scatter plot of the network embedding, with points
        colored by cluster membership if clustering has been performed.

        Returns
        -------
        None
            Displays the plot using matplotlib.

        Notes
        -----
        If embedding or clustering haven't been computed yet, this method will
        automatically call embed() and cluster_networks() first.

        Examples
        --------
        >>> app = Apparent()
        >>> app.pull("SELECT * FROM physician_data WHERE year >= 2020")
        >>> app.build_networks()
        >>> app.add_features(edge_features=["forman_curvature"])
        >>> app.plot_embedding()
        """
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
