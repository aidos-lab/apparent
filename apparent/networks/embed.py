class NetworkEmbedder:

    # TODO: Different methods for embedding networks kernels, or use pairwise distances from
    def __init__(self, network):
        self.network = network

    def embed(self, data):
        return self.network(data)
