import pytest
import pandas as pd
import networkx as nx


from apparent.apparent import Apparent
from apparent.networks.build import NetworkBuilder


class TestApparent:

    def test_fetcher(
        self,
        local_url,
        max_HSA_query,
        max_HSA_size,
    ):

        A = Apparent(base_url=local_url)

        data = A._fetcher(max_HSA_query)

        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] == max_HSA_size

    def test_pull(self, sample_query, local_url):
        A = Apparent(base_url=local_url)

        A.pull(sample_query)

        assert isinstance(A.data, pd.DataFrame)
        assert A.data.shape[0] == 1005
        assert A.data.shape[1] == 13

        assert isinstance(A.hsas, list)
        assert isinstance(A.years, list)

    def test_batch_interaction_queries(self, sample_query, local_url):
        A = Apparent(base_url=local_url)
        A.pull(sample_query)
        queries = A._batch_interaction_queries()

        assert isinstance(queries, list)
        assert len(queries) == len(A.hsas) * len(A.years)

    def test_download_interactions(
        self,
        local_url,
        sample_query,
    ):

        A = Apparent(base_url=local_url)
        A.pull(sample_query)
        A.download_interactions()

        assert isinstance(A.physician_interactions, pd.DataFrame)

    def test_build_networks(self, local_url, sample_query):
        A = Apparent(base_url=local_url)

        A.pull(sample_query)
        A.build_networks()

        assert isinstance(A.networks, dict)
        assert len(A.networks) == 5
        assert all(isinstance(v, nx.Graph) for v in A.networks.values())
        assert all(v.number_of_nodes() > 0 for v in A.networks.values())
        assert all(v.number_of_edges() > 0 for v in A.networks.values())
