import random

from torch.optim.lr_scheduler import ReduceLROnPlateau

from node2vec import *
from word2vec import *
import torch as t
from torch.utils.data import Dataset
import time


class EmbeddingAlignment(t.nn.Module):

    def __init__(self, dataloader, device, bpe, gamma, b):
        super().__init__()
        self.dataloader = dataloader
        self.index_mapping = self.dataloader.index_map
        self.title_emb, self.desc_emb = self.create_embedding_layers_word2vec()
        self.title_emb.to(device)
        self.desc_emb.to(device)
        self.node_emb = self.create_embedding_layer_node2vec().to(device)
        self.batches_per_epoch = bpe
        self.gamma = gamma
        self.b = b
        self.device = device
        # Cosine similarity across the first dimension
        self.cosine_similarity = t.nn.CosineSimilarity(dim=1)

    def create_embedding_layers_word2vec(self):
        title_emb = np.zeros((self.dataloader.max_n_index + 1, 300))
        desc_emb = np.zeros((self.dataloader.max_n_index + 1, 300))

        for asin, title, desc in zip(self.dataloader.attr_emb['asin'], self.dataloader.attr_emb['title_embed'],
                                     self.dataloader.attr_emb['description_embed']):
            # Here we do remove somes nodes that are no longer useful in the graph due to preprocessing
            if asin in self.index_mapping:
                title_emb[self.index_mapping[asin]] = title
                desc_emb[self.index_mapping[asin]] = desc
        weight_title = t.FloatTensor(title_emb)
        weight_desc = t.FloatTensor(desc_emb)

        title_emb_layer = t.nn.Embedding.from_pretrained(weight_title)
        desc_emb_layer = t.nn.Embedding.from_pretrained(weight_desc)
        title_emb_layer.weight.requires_grad = True
        desc_emb_layer.weight.requires_grad = True

        return title_emb_layer, desc_emb_layer

    def create_embedding_layer_node2vec(self):
        node_emb = np.zeros((self.dataloader.max_n_index + 1, 300))

        graph_used_set = set(self.dataloader.graph.flatten())
        new_index = 0
        reduced_index_mapping = {}
        for i, (key, value) in enumerate(self.index_mapping.items()):
            if value in graph_used_set:
                reduced_index_mapping[key] = new_index
                new_index += 1

        for asin, idx in self.index_mapping.items():
            node_emb[idx] = self.dataloader.node_emb.wv.get_vector(idx)

        node_emb_tensor = t.FloatTensor(node_emb)
        node_emb_layer = t.nn.Embedding.from_pretrained(node_emb_tensor)
        node_emb_layer.weight.requires_grad = True
        return node_emb_layer

    def save_embedding_layers(self):
        model_folder = Path(__file__).parent.parent / 'model'

        layers = {'desc_emb': self.desc_emb, 'title_emb': self.title_emb, 'node_emb': self.node_emb}
        t.save(layers, model_folder/'aligned_emb')

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

    def alignment_loss(self, attr_emb, node_emb, links):
        num_pairs = attr_emb.shape[1]
        neg_part = (1 - links) * self.logistic_loss(self.cosine_similarity(attr_emb[0], node_emb[1]).neg())
        pos_part = links * self.logistic_loss(self.cosine_similarity(attr_emb[0], node_emb[1]))
        return t.sum((neg_part + pos_part)) / num_pairs

    def total_loss(self, batch, links, gamma_1, gamma_2, gamma_3):
        desc_emb = self.desc_emb(batch)

        title_emb = self.title_emb(batch)
        attr_emb = t.mean(t.stack((desc_emb, title_emb), dim=2), dim=2)
        node_emb = self.node_emb(batch)

        alignment_loss = self.alignment_loss(attr_emb, node_emb, links)
        rank_loss_attr = self.rank_loss(attr_emb, links, 1, 2)
        rank_loss_nodes = self.rank_loss(node_emb, links, 1, 2)
        loss = gamma_1 * rank_loss_attr + gamma_2 * rank_loss_nodes + gamma_3 * alignment_loss
        return -loss, alignment_loss

    def weight_function(self, beta, const):
        """
        Function to weigh samples, doesn't do anything now. Should use shortest path, but that is VERY infeasible for
        large graphs due to > O(N^2) complexity to calculate all shortest paths.
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

    def train_epoch(self, optimizer, device):
        running_alignment_loss = 0.
        running_loss = 0.

        for i in range(self.batches_per_epoch):
            optimizer.zero_grad()

            nodes, y = self.dataloader[i]
            nodes = nodes.to(device)
            y = y.to(device)

            loss, alignment_loss = self.total_loss(nodes, y, 1, 1, 1)
            loss.backward()

            optimizer.step()

            running_alignment_loss += alignment_loss.item()
            running_loss += loss.item()

        avg_loss_align_loss = running_alignment_loss / self.batches_per_epoch
        avg_loss = running_loss / self.batches_per_epoch
        # print("Average Loss: {}".format(avg_loss))
        return avg_loss_align_loss, avg_loss


class NodePairBatchLoader(Dataset):
    def __init__(self, file_path, file_path_index, file_path_attr, file_path_node, file_path_sparse, batch_size):
        super().__init__()
        self.file = file_path
        self.file_index_map = file_path_index
        self.file_attr_emb = file_path_attr
        self.file_node_emb = file_path_node
        self.file_sparse_adj = file_path_sparse

        # Load files for alignment
        self.graph = self.load_graph_file()
        self.index_map = self.load_index_map()
        self.attr_emb = self.load_attr_emb()
        self.node_emb = self.load_node_emb()
        self.sparse_adj_matrix = self.load_sparse_adj_matrix()

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
        y = np.zeros(self.batch_size)

        for i, (start, end) in enumerate(zip(nodes[0], nodes[1])):
            if self.sparse_adj_matrix[start, end] == 1:
                y[i] = 1

        # Shuffle the batch to ensure random distribution of y=1's
        shuffle_index = np.random.rand(y.shape[0]).argsort()
        nodes_shuffled = np.take(nodes, shuffle_index, axis=1)
        y_shuffled = np.take(y, shuffle_index, axis=0)
        y = t.IntTensor(y_shuffled)
        nodes = t.IntTensor(nodes_shuffled)
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

    def load_sparse_adj_matrix(self):
        with open(self.file_sparse_adj, "rb") as f:
            return pickle.load(f)

    def get_batch(self):
        # This can generate self edge connections, but should happen very rarely
        nodes = np.random.randint(0, self.n_nodes, (2, self.batch_size - int(self.batch_size * .1)))
        real_connection = self.graph[:, np.random.randint(0, self.n_edges, size=int(self.batch_size * .1))]
        nodes = np.c_[nodes, real_connection]
        return nodes


def train_model(num_epochs):
    data_folder = Path(__file__).parent.parent / 'data'
    data_loader = NodePairBatchLoader(data_folder / "connectivity_array.npy", data_folder / "index_mapping.pkl",
                                      data_folder / "embed_data.gzip", data_folder / "node2vec_model.pkl",
                                      data_folder / 'sparse_adj_matrix.pkl', 500)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingAlignment(data_loader, device, 1000, 1, 1)
    model.to(device)

    optimizer = t.optim.Adam(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    model.train()
    for i in range(num_epochs):
        avg_align_loss, avg_loss_epoch = model.train_epoch(optimizer, device)
        scheduler.step(avg_loss_epoch)
        print("Average Loss: {}, average alignment loss: {}, epoch {}/{}".format(avg_loss_epoch, avg_align_loss,
                                                                                 i+1, num_epochs))
    model.save_embedding_layers()

if __name__ == "__main__":
    # data_path = Path(__file__).parent.parent / 'data'
    # loader = NodePairBatchLoader(data_path / "connectivity_array.npy", data_path / "index_mapping.pkl",
    #                              data_path / "embed_data.gzip", data_path / "node2vec_model.pkl", 10)
    # # node_attr = pd.read_parquet(data_path / 'embed_data.gzip')
    # # print(node_attr)
    # embedding_align = EmbeddingAlignment(loader, 5, 10, 1)
    # embedding_align.train_epoch()
    train_model(20)
