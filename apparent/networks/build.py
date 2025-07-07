import networkx as nx
import pandas as pd
from typing import Callable, Optional, Dict


class NetworkBuilder:
    """
    A class for building network graphs from edge data, with support for custom graph-building recipes.

    This class provides a flexible framework for constructing NetworkX graphs from
    edge data. It supports both a default standard build method and custom user-defined
    build functions.

    Parameters
    ----------
    custom_build : callable, optional
        A custom function that accepts the dataframe and additional keyword arguments
        (`**kwargs`) and returns a NetworkX graph. If not provided, the default
        `standard_build` method is used.

    Attributes
    ----------
    build_recipe : callable
        The function used to build networks, either custom_build or standard_build.

    Examples
    --------
    >>> import pandas as pd
    >>> from apparent.networks import NetworkBuilder
    >>> 
    >>> # Create sample edge data
    >>> edges_df = pd.DataFrame({
    ...     'npi_a': [1, 2, 3],
    ...     'npi_b': [2, 3, 1],
    ...     'hsa': [100, 100, 100],
    ...     'year': [2020, 2020, 2020],
    ...     'a2b': [5, 3, 2],
    ...     'b2a': [2, 1, 4]
    ... })
    >>> 
    >>> # Build network using default method
    >>> builder = NetworkBuilder()
    >>> graph = builder.build(edges_df, hsa=100, year=2020)
    >>> print(f"Network has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
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
            A function that accepts `edges_df` (pandas DataFrame) and keyword arguments
            (`**kwargs`) as input, and returns a NetworkX graph (nx.Graph). The function
            signature should be: func(edges_df, **kwargs) -> nx.Graph. If not provided,
            the default `standard_build` method will be used.

        Examples
        --------
        >>> def custom_builder(edges_df, **kwargs):
        ...     # Custom logic here
        ...     return nx.Graph()
        >>> 
        >>> builder = NetworkBuilder(custom_build=custom_builder)
        """
        # Use the provided custom build recipe, or default to the standard build if none is provided
        self.build_recipe = custom_build or self.standard_build

    def standard_build(
        self, edges_df: pd.DataFrame, hsa: int, year: int
    ) -> nx.Graph:
        """
        Default network-building method that creates a NetworkX graph from edge data.

        This method filters the input DataFrame by the specified HSA and year, then
        constructs an undirected NetworkX graph using the edge list. Edge attributes
        'a2b' and 'b2a' are preserved in the resulting graph.

        Parameters
        ----------
        edges_df : pd.DataFrame
            A pandas DataFrame containing the edges with columns `npi_a`, `npi_b`,
            `hsa`, `year`, `a2b`, and `b2a`.
        hsa : int
            The Health Service Area (HSA) code to filter the network.
        year : int
            The year to filter the network.

        Returns
        -------
        nx.Graph
            An undirected NetworkX graph built from the filtered edge data.
            Nodes are identified by NPI codes, and edges contain 'a2b' and 'b2a' attributes.

        Examples
        --------
        >>> builder = NetworkBuilder()
        >>> graph = builder.standard_build(edges_df, hsa=100, year=2020)
        >>> print(list(graph.edges(data=True)))
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

        This method serves as the main entry point for building networks. It delegates
        to either the standard_build method or a custom build function depending on
        how the NetworkBuilder was initialized.

        Parameters
        ----------
        edges_df : pd.DataFrame
            A pandas DataFrame containing the edges with columns `npi_a`, `npi_b`,
            `hsa`, `year`, `a2b`, and `b2a`.
        **kwargs : dict
            Additional keyword arguments that can be passed to the build function.
            For standard_build, this typically includes 'hsa' and 'year' parameters.

        Returns
        -------
        nx.Graph
            The NetworkX graph constructed using the selected recipe.

        Examples
        --------
        >>> builder = NetworkBuilder()
        >>> graph = builder.build(edges_df, hsa=100, year=2020)
        >>> print(f"Built network with {graph.number_of_nodes()} nodes")
        """
        return self.build_recipe(edges_df, **kwargs)
