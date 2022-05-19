import random
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch as t
from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.embedding_alignment import NodePairBatchLoader


def split_graph(graph_edges, index_mapping):
    random.seed(2022)
    id_asin_mapping = {value: key for (key, value) in index_mapping.items()}

    val_nodes = random.sample(list(index_mapping.keys()), int(len(index_mapping) * .1))
    val_nodes_dict = {val_node: index_mapping[val_node] for val_node in val_nodes}
    id_asin_val = {index_mapping[val_node]: val_node for val_node in val_nodes}

    index_map_red = {key: value for key, value in index_mapping.items() if key not in val_nodes_dict}
    id_asin_red = {value: key for key, value in index_mapping.items() if key not in val_nodes_dict}

    test_nodes = random.sample(list(index_map_red.keys()), int(len(index_mapping) * .1))
    test_nodes_dict = {test_node: index_mapping[test_node] for test_node in test_nodes}
    id_asin_test = {index_mapping[test_node]: test_node for test_node in test_nodes}

    index_map_train = {key: value for key, value in index_map_red.items() if key not in test_nodes_dict}
    id_asin_train = {value: key for key, value in index_map_train.items()}

    train_mask = [True if start in id_asin_train and end in id_asin_train else False
                  for (start, end) in zip(graph_edges[0], graph_edges[1])]
    val_mask = [True if start in id_asin_val and end in id_asin_val else False
                for (start, end) in zip(graph_edges[0], graph_edges[1])]
    test_mask = [True if start in id_asin_test and end in id_asin_test else False
                 for (start, end) in zip(graph_edges[0], graph_edges[1])]
    graph_edges_train = graph_edges[:, train_mask]
    graph_edges_val = graph_edges[:, val_mask]
    graph_edges_test = graph_edges[:, test_mask]

    return graph_edges_train, graph_edges_val, graph_edges_test, index_map_train, val_nodes_dict, test_nodes_dict


class LinkPrediction(t.nn.Module):
    def __init__(self, loss):
        super(LinkPrediction, self).__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

        self.train_dataloader = NodePairBatchLoader(data_path / "graph_edges_arrays/graph_train.npy",
                                                    data_path / "index_mappings/index_map_train.pkl",
                                                    data_path / "embed_data.gzip", data_path / "node2vec_model.pkl",
                                                    data_path / 'sparse_adj_matrix.pkl', 500)
        self.optimizer = t.optim.Adam(self.parameters(), lr=0.1)
        self.loss = loss

        self.title_emb, self.desc_emb, self.node_emb = self.load_emb_test()

        self.input_dense_node = t.nn.Linear(600, 100, device=self.device)
        self.input_dense_attr = t.nn.Linear(600, 100, device=self.device)
        self.hidden_layer_node = t.nn.Linear(100, 25, device=self.device)
        self.hidden_layer_attr = t.nn.Linear(100, 25, device=self.device)
        self.output_value_node = t.nn.Linear(25, 1, device=self.device)
        self.output_value_attr = t.nn.Linear(25, 1, device=self.device)
        self.activation = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()

        self.dropout = t.nn.Dropout(p=.1)
        self.batch_norm1 = t.nn.BatchNorm1d(600, device=self.device)

    def forward(self, inputs_node, inputs_attr, gamma):
        x_node = self.relu(self.input_dense_node(inputs_node))
        x_node = self.batch_norm1(x_node)
        x_node = self.relu(self.hidden_layer_node(x_node))
        x_node = self.dropout(x_node)
        x_node = self.output_value_node(x_node)

        x_attr = self.relu(self.input_dense_attr(inputs_attr))
        x_attr = self.batch_norm1(x_attr)
        x_attr = self.relu(self.hidden_layer_attr(x_attr))
        x_attr = self.dropout(x_attr)
        x_attr = self.output_value_attr(x_attr)

        return (gamma[0] * x_node + gamma[1] * x_attr) / t.sum(gamma)

    def get_loss(self, pred_y, y):
        return self.loss(pred_y, y)

    def train_epoch(self, optimizer):
        running_loss = 0.

        for i in range(self.batches_per_epoch):
            optimizer.zero_grad()

            nodes, y = self.dataloader[i]
            nodes = nodes.to(self.device)
            y = y.to(self.device)

            test = self.title_emb(nodes)


            pred_y = self.forward()
            loss, alignment_loss = self.get_loss()
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / self.batches_per_epoch
        # print("Average Loss: {}".format(avg_loss))
        return avg_loss

    def load_emb_test(self):
        embeddings = t.load(Path(__file__).parent.parent / 'model' / 'aligned_emb')
        embeddings['title_emb'].trainable = False
        embeddings['desc_emb'].trainable = False
        embeddings['node_emb'].trainable = False
        return embeddings['title_emb'], embeddings['desc_emb'], embeddings['node_emb']

def train_model():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = LinkPrediction(t.nn.BCEWithLogitsLoss()).to(device)

    optimizer = t.optim.Adam(model.parameters(), lr=0.1)
    model.train_epoch(optimizer)



class LinkPredictionLoader(Dataset):
    def __init__(self, file_path_graph, file_path_index, file_path_sparse, batch_size):
        super().__init__()
        self.file_graph = file_path_graph
        self.file_index_map = file_path_index
        self.file_sparse_adj = file_path_sparse

        # Load files for alignment
        self.graph = self.load_graph_file()
        self.index_map = self.load_index_map()
        self.sparse_adj_matrix = self.load_sparse_adj_matrix()
        # self.attr_emb, self.node_emb = self.load_emb()
        self.title_emb, self.desc_emb, self.node_emb = self.load_emb_test()

        # Pre-calculate some useful statistics
        self.n_nodes = len(self.index_map)
        self.max_n_index = max(self.index_map.values())
        self.n_edges = self.graph.shape[1]
        self.batch_size = batch_size
        pass

    def __getitem__(self, idx):
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
        graph = np.load(self.file_graph).astype(int)
        return graph

    def load_index_map(self):
        with open(self.file_index_map, 'rb') as f:
            return pickle.load(f)

    def load_emb(self):
        embeddings = t.load(Path(__file__).parent.parent / 'model' / 'aligned_emb')
        attr_emb = np.mean(np.array([np.array(embeddings['title_emb'].weight.data.cpu()),
                                     np.array(embeddings['desc_emb'].weight.data.cpu())]), axis=0)
        node_emb = np.array(embeddings['node_emb'].weight.data.cpu())
        return attr_emb, node_emb

    def load_emb_test(self):
        embeddings = t.load(Path(__file__).parent.parent / 'model' / 'aligned_emb')
        return embeddings['title_emb'], embeddings['desc_emb'], embeddings['node_emb']

    def load_sparse_adj_matrix(self):
        with open(self.file_sparse_adj, "rb") as f:
            return pickle.load(f)

    def get_batch(self):
        # This can generate self edge connections, but should happen very rarely
        nodes = np.random.choice(list(self.index_map.values()), (2, self.batch_size - int(self.batch_size * .1)))
        real_connection = self.graph[:, np.random.randint(0, self.n_edges, size=int(self.batch_size * .1))]
        nodes_full = np.c_[nodes, real_connection]
        return nodes_full


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data'

    # with open(data_path / 'index_mapping.pkl', 'rb') as f:
    #     index_map = pickle.load(f)
    # edge_array = np.load(str(data_path / 'connectivity_array.npy'))

    # graph_t, graph_v, graph_test, index_map_t, index_map_v, index_map_test = split_graph(edge_array, index_map)
    # test = LinkPredictionLoader(data_path / "graph_edges_arrays/graph_train.npy",
    #                             data_path / "index_mappings/index_map_train.pkl",
    #                             data_path / 'sparse_adj_matrix.pkl', 100)
    # test.get_batch()
    # np.save(data_path / "graph_edges_arrays/graph_train.npy", graph_t)
    # np.save(data_path / "graph_edges_arrays/graph_val.npy", graph_v)
    # np.save(data_path / "graph_edges_arrays/graph_test.npy", graph_test)
    # with open(data_path / "index_mappings/index_map_train.pkl", "wb") as f:
    #     pickle.dump(index_map_t, f)
    # with open(data_path / "index_mappings/index_map_val.pkl", "wb") as f:
    #     pickle.dump(index_map_v, f)
    # with open(data_path / "index_mappings/index_map_test.pkl", "wb") as f:
    #     pickle.dump(index_map_test, f)

    train_model()

