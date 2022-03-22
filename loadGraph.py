import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, coo_matrix
from tqdm import tqdm
import torch
from torch_geometric.data import Data

# https://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence


def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def load_graph():
    with gzip.open("data/metadata_books.json") as f:
        products = []; product_list = []; index_map = {}
        total_edges = 0
        for i, line in enumerate(f):
            entry = json.loads(line.strip())
            products.append(entry['asin']); product_list.append(entry['also_buy'])
            # Creating hash map of products for easy lookup
            index_map[entry['asin']] = i
            # Calculating total number of edges to create edge matrix
            total_edges += len(entry['also_buy'])
        # adjacency_matrix = dok_matrix((len(products), len(products)))

        # Connectivity matrix, which has two entries for each of the nodes connected to the edge like
        # [1, 0, 3], [2, 3, 2] denotes edges 1 to 2, 0 to 3 and 3 to 2.
        graph_connectivity = np.zeros(shape=(2, total_edges))
        edge_index = 0

        with tqdm(total=len(products)) as pbar:
            for j, product_list in enumerate(product_list):
                for product in product_list:
                    if product in index_map:
                        graph_connectivity[0][edge_index] = j
                        graph_connectivity[1][edge_index] = index_map[product]
                        edge_index += 1
                pbar.update()

        # with tqdm(total=len(products)) as pbar:
        #     for j, product_list in enumerate(product_list):
        #         for product in product_list:
        #             if product in index_map:
        #                 adjacency_matrix[j,index_map[product]] = 1
        #                 # adjacency_matrix[index_map[product],j] = 1
        #         pbar.update()
        # for i in range(len(products)):
        #     if adjacency_matrix[i,:].count_nonzero() == 0 and adjacency_matrix[:,i].count_nonzero() == 0:
        #
        #
        # print(adjacency_matrix.count_nonzero())
    return graph_connectivity


if __name__ == "__main__":
    graph_matrix = torch.tensor(load_graph())
    data = Data(edge_index=graph_matrix)
    torch.save(data, "data/graph_object_no_features")
