import networkx as nx
import pandas as pd
from typing import Callable, Optional, Dict


class NetworkBuilder:
    """
    A class for building network graphs from edge data, with support for custom graph-building recipes.

    Parameters
    ----------
    custom_build : callable, optional
        A custom function that accepts the dataframe and additional keyword arguments (`**kwargs`)
        and returns a NetworkX graph. If not provided, the default `standard_build` method is used.

    Methods
    -------
    build(edges_df, **kwargs)
        Builds a network graph using the selected recipe (default or custom).
    """

    def __init__(
        self,
        custom_build: Optional[Callable[[pd.DataFrame, Dict], nx.Graph]] = None,
    ):
        """
        Initialize the NetworkBuilder with a custom build recipe.

        Parameters
        ----------
        custom_build : callable, optional
            A function that accepts `edges_df` (pandas DataFrame) and keyword arguments (`**kwargs`)
            as input, and returns a NetworkX graph (nx.Graph). If not provided, the default `standard_build`
            method will be used.
        """
        # Use the provided custom build recipe, or default to the standard build if none is provided
        self.build_recipe = custom_build or self.standard_build

    def standard_build(
        self, edges_df: pd.DataFrame, hsa: int, year: int
    ) -> nx.Graph:
        """
        Default network-building method that creates a NetworkX graph from edge data.

        Parameters
        ----------
        edges_df : pd.DataFrame
            A pandas DataFrame containing the edges with columns `npi_a`, `npi_b`,
            `hsa`, `year`, `a2b`, and `b2a`.
        hsa : int
            The region code (Hsa) to filter the network.
        year : int
            The year to filter the network.

        Returns
        -------
        nx.Graph
            An undirected NetworkX graph built from the edge data.
        """
        # Initialize an undirected graph G
        G = nx.Graph()

        # Populate G using the filtered DataFrame
        G = nx.from_pandas_edgelist(
            df=edges_df[(edges_df.hsa == hsa) & (edges_df.year == year)],
            source="npi_a",
            target="npi_b",
            edge_attr=["a2b", "b2a"],
        )

        return G

    def build(self, edges_df: pd.DataFrame, **kwargs) -> nx.Graph:
        """
        Build the network using the selected recipe (default or custom).

        Parameters
        ----------
        edges_df : pd.DataFrame
            A pandas DataFrame containing the edges with columns `npi_a`, `npi_b`,
            `hsa`, `year`, `a2b`, and `b2a`.
        **kwargs : dict
            Additional keyword arguments that can be passed to the custom build function.

        Returns
        -------
        nx.Graph
            The NetworkX graph constructed using the selected recipe.
        """
        return self.build_recipe(edges_df, **kwargs)
