import pytest
import pandas as pd
import networkx as nx
import tempfile
import os
from unittest.mock import patch


from apparent.apparent import Apparent


class TestApparent:
    @pytest.mark.unit
    def test_init_before_pull(self, apparent_url):
        """Test the initial state of Apparent object before pulling data."""
        A = Apparent(base_url=apparent_url)
        
        # Check initial state
        assert A.base_url == apparent_url
        assert A.data is None
        assert A.kilt is None
        assert A.networks == {}
        assert A.distances == {}
        for attr in [
            'network_ids', 'physician_interactions', 'builder',
            'comparator', 'embedder', 'clusterer', 'embedding'
]:
            assert attr not in A.__dict__

    @pytest.mark.integration
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

    @pytest.mark.integration
    def test_pull(self, sample_query, apparent_url, num_networks):
        A = Apparent(base_url=apparent_url)

        A.pull(sample_query)

        assert isinstance(A.data, pd.DataFrame)
        assert A.data.shape[0] == num_networks
        assert A.data.shape[1] == 13

        assert isinstance(A.network_ids, list)
        assert len(A.network_ids) == num_networks

    @pytest.mark.unit
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
    def test_compare(self, local_url, sample_query):
        A = Apparent(base_url=local_url)
        A.pull(sample_query)
        A.build_networks()
        A.compare(measure="forman_curvature")

    @pytest.mark.integration
    def test_embed(self, local_url, sample_query):
        A = Apparent(base_url=local_url)
        A.pull(sample_query)
        A.build_networks()
        A.embed()

        assert hasattr(A, "embedding")
        assert A.embedding.shape[1] == 2

    @pytest.mark.integration
    def test_cluster(self, local_url, sample_query):
        A = Apparent(base_url=local_url)
        A.pull(sample_query)
        A.build_networks()
        A.cluster_networks()

    @pytest.mark.unit
    def test_pull_missing_hsa_year(self, apparent_url):
        """
        Test that Apparent.pull() handles missing 'hsa' or 'year' columns in query results.
        """
        A = Apparent(base_url=apparent_url)

        # Case 1: Missing both 'hsa' and 'year'
        with patch.object(
            A, "_fetcher", return_value=pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        ) as mock_fetcher:
            with pytest.raises(ValueError) as excinfo:
                A.pull("SELECT foo, bar FROM table")
            assert "must contain 'hsa' and 'year' columns" in str(excinfo.value)

        # Case 2: Missing only 'hsa'
        with patch.object(
            A, "_fetcher", return_value=pd.DataFrame({"year": [2020, 2021], "foo": [1, 2]})
        ) as mock_fetcher:
            with pytest.raises(ValueError) as excinfo:
                A.pull("SELECT year, foo FROM table")
            assert "must contain 'hsa' and 'year' columns" in str(excinfo.value)

        # Case 3: Missing only 'year'
        with patch.object(
            A, "_fetcher", return_value=pd.DataFrame({"hsa": [1, 2], "foo": [3, 4]})
        ) as mock_fetcher:
            with pytest.raises(ValueError) as excinfo:
                A.pull("SELECT hsa, foo FROM table")
            assert "must contain 'hsa' and 'year' columns" in str(excinfo.value)

    @pytest.mark.unit
    def test_download_interactions_without_data(self, capfd):
        """
        Test that download_interactions returns None and prints a warning if called before pulling data.
        """
        A = Apparent(base_url="http://dummy-url")
        result = A.download_interactions()
        out, err = capfd.readouterr()
        assert result is None
        assert "No data available. Please fetch data first using `Apparent.pull()`." in out

    @pytest.mark.unit
    def test_build_networks_calls_download_when_physician_interactions_none(self):
        """
        Test that build_networks() calls download_interactions() when physician_interactions is None.
        """
        A = Apparent(base_url="http://dummy-url")
        
        # Set physician_interactions to None (simulating failed download_interactions)
        A.physician_interactions = None
        
        # Mock download_interactions to track if it's called
        with patch.object(A, 'download_interactions') as mock_download:
            # Mock the physician_interactions to avoid the AttributeError 
            mock_download.side_effect = lambda: setattr(A, 'physician_interactions', pd.DataFrame({'hsa': [1], 'year': [2020]}))
            
            # This should call download_interactions because physician_interactions is None
            try:
                A.build_networks()
            except Exception:
                # We expect this to fail due to missing NetworkBuilder import, but download_interactions should still be called
                pass
                
        # Verify download_interactions was called
        mock_download.assert_called_once()

    @pytest.mark.unit  
    def test_build_networks_calls_download_when_physician_interactions_missing(self):
        """
        Test that build_networks() calls download_interactions() when physician_interactions attribute doesn't exist.
        """
        A = Apparent(base_url="http://dummy-url")
        
        # Ensure physician_interactions attribute doesn't exist
        if hasattr(A, 'physician_interactions'):
            delattr(A, 'physician_interactions')
        
        # Mock download_interactions to track if it's called
        with patch.object(A, 'download_interactions') as mock_download:
            # Mock the physician_interactions to avoid the AttributeError
            mock_download.side_effect = lambda: setattr(A, 'physician_interactions', pd.DataFrame({'hsa': [1], 'year': [2020]}))
            
            # This should call download_interactions because physician_interactions doesn't exist
            try:
                A.build_networks()
            except Exception:
                # We expect this to fail due to missing NetworkBuilder import, but download_interactions should still be called
                pass
                
        # Verify download_interactions was called
        mock_download.assert_called_once()