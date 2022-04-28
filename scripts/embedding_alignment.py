import random

from node2vec import *
from word2vec import *
import torch as t
from torch.utils.data import Dataset


class EmbeddingAlignment(t.nn.Module):

    def __init__(self, dataloader, bpe, gamma, b):
        super().__init__()
        self.dataloader = dataloader
        self.index_mapping = self.dataloader.index_map
        self.title_emb, self.desc_emb = self.create_embedding_layers_word2vec()
        self.node_emb = self.create_embedding_layer_node2vec()
        self.batches_per_epoch = bpe
        self.gamma = gamma
        self.b = b
        # Cosine similarity across the first dimension
        self.cosine_similarity = t.nn.CosineSimilarity(dim=1)

    def create_embedding_layers_word2vec(self):
        title_emb = np.zeros((self.dataloader.max_n_index+1, 300))
        desc_emb = np.zeros((self.dataloader.max_n_index+1, 300))

        for asin, title, desc in zip(self.dataloader.attr_emb['asin'], self.dataloader.attr_emb['title_embed'],
                            self.dataloader.attr_emb['description_embed']):
            title_emb[self.index_mapping[asin]] = title
            desc_emb[self.index_mapping[asin]] = desc
        weight_title = t.FloatTensor(title_emb)
        weight_desc = t.FloatTensor(desc_emb)

        title_emb_layer = t.nn.Embedding.from_pretrained(weight_title)
        desc_emb_layer = t.nn.Embedding.from_pretrained(weight_desc)

        return title_emb_layer, desc_emb_layer

    def create_embedding_layer_node2vec(self):
        node_emb = np.zeros((self.dataloader.max_n_index+1, 300))

        for asin in self.dataloader.attr_emb['asin']:
            node_emb[self.index_mapping[asin]] = self.dataloader.node_emb.wv.get_vector(self.index_mapping[asin])

        # Check this, might nog have aligned nodes and attributes
        for i, org_id in enumerate(set(self.index_mapping.values())):

            node_emb[i] = self.dataloader.node_emb.wv.get_vector(i)

        node_emb_tensor = t.FloatTensor(node_emb)
        node_emb_layer = t.nn.Embedding.from_pretrained(node_emb_tensor)
        return node_emb_layer

    def rank_loss(self, batch, links, gamma, beta):
        """

        :param batch: Batch of node pairs (B), with shape [num_pairs, 2, embedding_size)
        :param links: Batch of variable indicating if node pairs are neighbours (y_i), with shape [num_pairs, 1)
        :param gamma: Hyper parameter for logistic loss
        :param beta: Hyper parameter for logistic loss
        :return: The rank_loss for one batch (might want to do all batches at once?)
        """
        num_pairs = batch.shape[1]
        neg_part = (1 - links) * self.logistic_loss(self.cosine_similarity(batch[0], batch[1]).neg())
        pos_part = links * self.logistic_loss(self.cosine_similarity(batch[0], batch[1]))
        return t.sum((neg_part + pos_part)) / num_pairs

    def total_loss(self, batch, feature_batch):
        pass

    def weight_function(self, beta, const):
        """
        Function to weigh samples, doesn't do anything now. Should use shortest path, but that is VERY infeasible for
        large graphs due to O(N^2) complexity to calculate all shortest paths.
        :param beta:
        :param const:
        :return:
        """
        # We deviate from the implantation of paper due to computational cost of calculating every shortest path
        return t.exp(beta / const)

    def logistic_loss(self, x):
        """
        Logistic loss function used in rank_loss
        :param x: Input values, usually a vector of cosine similarities or dot products, shape [num_pairs, 1]
        :return: Logistic loss value vector
        """
        return (1 / self.gamma) * t.log(1 + t.exp(-self.gamma * x + self.b))

    def train_epoch(self):
        for i in range(self.batches_per_epoch):
            nodes, y = self.dataloader[i]
            attr_emb = self.desc_emb(nodes)
            test = self.rank_loss(attr_emb, y, 5, 5)

        pass


class NodePairBatchLoader(Dataset):
    def __init__(self, file_path, file_path_index, file_path_attr, file_path_node, batch_size):
        super().__init__()
        self.file = file_path
        self.file_index_map = file_path_index
        self.file_attr_emb = file_path_attr
        self.file_node_emb = file_path_node

        # Load files for alignment
        self.graph = self.load_graph_file()
        self.index_map = self.load_index_map()
        self.attr_emb = self.load_attr_emb()
        self.node_emb = self.load_node_emb()

        # Pre-calculate some useful statistics
        self.n_nodes = len(set(self.graph.flatten()))
        self.max_n_index = max(self.index_map.values())
        self.n_edges = self.graph.shape[1]
        self.batch_size = batch_size
        pass

    def __getitem__(self, idx):
        # We can speed up training quick and dirty by skipping this step and assuming that every randomly generated
        # Batch is not a connection. We should do some probability calculations to check what the probability of
        # Randomly selecting a connection is.
        nodes = self.get_batch()
        start_nodes = self.graph[0]
        end_nodes = self.graph[1]
        y = np.zeros(self.batch_size)
        # I should make this vectorized I guess
        for i, (start, end) in enumerate(zip(nodes[0], nodes[1])):
            links = np.logical_and((end_nodes == end), (start_nodes == start))
            if links.any():
                y[i] = 1
        y = t.IntTensor(y)
        return nodes, y

    def load_graph_file(self):
        graph = np.load(self.file).astype(int)
        return graph

    def load_index_map(self):
        with open(self.file_index_map, 'rb') as f:
            return pickle.load(f)

    def load_attr_emb(self):
        node_attr = pd.read_parquet(self.file_attr_emb)
        return node_attr

    def load_node_emb(self):
        with open(self.file_node_emb, "rb") as f:
            return pickle.load(f)

    def get_batch(self):
        # This can generate self edge connections, but should happen very rarely
        nodes = np.random.randint(0, self.n_nodes, (2, self.batch_size - 1))
        real_connection = self.graph[:, np.random.randint(self.n_edges)]
        nodes = t.IntTensor(np.c_[nodes, real_connection])
        return nodes


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data'
    loader = NodePairBatchLoader(data_path / "connectivity_array.npy", data_path / "index_mapping.pkl",
                                 data_path / "embed_data.gzip", data_path / "node2vec_model.pkl", 10)

    # node_attr = pd.read_parquet(data_path / 'embed_data.gzip')
    # print(node_attr)
    embedding_align = EmbeddingAlignment(loader, 5, 10, 1)
    embedding_align.train_epoch()
