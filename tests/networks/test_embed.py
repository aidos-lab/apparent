import pytest
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from apparent.networks.embed import NetworkEmbedder

@pytest.mark.unit
class TestNetworkEmbedder:

    def test_defaults(self):
        embedder = NetworkEmbedder()
        assert embedder.D is None
        assert embedder.method == TSNE
        assert isinstance(embedder.scaler, StandardScaler)
        assert embedder.kwargs == {}

    def test_embed_with_pairwise_distances(self, random_distance_matrix):
        embedder = NetworkEmbedder(pairwise_distances=random_distance_matrix)
        embedding = embedder.embed()
        assert embedding.shape == (len(random_distance_matrix), 2)

    def test_embed_with_data(self, random_data):
        embedder = NetworkEmbedder()
        embedding = embedder.embed(data=random_data)
        assert embedding.shape == (30, 2)

    def test_embed_raises_value_error(self):
        embedder = NetworkEmbedder()
        with pytest.raises(ValueError):
            embedder.embed()
