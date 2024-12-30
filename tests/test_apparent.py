import pytest
import pandas as pd
import networkx as nx


from apparent.apparent import Apparent


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

    def test_pull(self, sample_query, apparent_url, num_networks):
        A = Apparent(base_url=apparent_url)

        A.pull(sample_query)

        assert isinstance(A.data, pd.DataFrame)
        assert A.data.shape[0] == num_networks
        assert A.data.shape[1] == 13

        assert isinstance(A.network_ids, list)
        assert len(A.network_ids) == num_networks

    def test_batch_interaction_queries(
        self,
        sample_query,
        apparent_url,
        num_networks,
    ):
        A = Apparent(base_url=apparent_url)
        A.pull(sample_query)
        queries = A._batch_interaction_queries()

        assert isinstance(queries, list)
        assert len(queries) == num_networks

    def test_download_interactions(
        self,
        local_url,
        sample_query,
    ):

        A = Apparent(base_url=local_url)
        A.pull(sample_query)
        A.download_interactions()

        assert isinstance(A.physician_interactions, pd.DataFrame)
        assert A.physician_interactions.shape[0] > 0

    def test_build_networks(self, local_url, sample_query, num_networks):
        A = Apparent(base_url=local_url)

        A.pull(sample_query)
        A.build_networks()

        assert isinstance(A.networks, dict)
        assert len(A.networks) == num_networks
        assert all(isinstance(v, nx.Graph) for v in A.networks.values())
        assert all(v.number_of_nodes() > 0 for v in A.networks.values())
        assert all(v.number_of_edges() > 0 for v in A.networks.values())

        assert "Networks" in A.data.columns

        assert A.data["Networks"].nunique() == num_networks
        for G in A.data.Networks:
            assert isinstance(G, nx.Graph)

    def test_add_features(
        self,
        local_url,
        sample_query,
        test_node_features,
        test_edge_features,
    ):
        A = Apparent(base_url=local_url)

        A.pull(sample_query)
        A.build_networks()
        A.add_features(
            node_features=test_node_features, edge_features=test_edge_features
        )

    def test_compare(self, local_url, sample_query):
        A = Apparent(base_url=local_url)
        A.pull(sample_query)
        A.build_networks()
        A.compare(measure="forman_curvature")

    def test_embed(self, local_url, sample_query):
        A = Apparent(base_url=local_url)
        A.pull(sample_query)
        A.build_networks()
        A.embed()

        assert hasattr(A, "embedding")
        assert A.embedding.shape[1] == 2

    def test_cluster(self, local_url, sample_query):
        A = Apparent(base_url=local_url)
        A.pull(sample_query)
        A.build_networks()
        A.cluster_networks()
