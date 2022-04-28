import gzip
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import dok_matrix, coo_matrix
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from fastnode2vec import Graph, Node2Vec

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


def load_graph(path):
    with gzip.open(path / 'meta_Books.json.gz') as f:
        products = []; co_purchase_list = []; index_map = {}
        linked_products = set()

        total_edges = 0
        product_number = 0
        print("Reading the data file.")
        for line in tqdm(f, total=2934949):
            entry = json.loads(line.strip())
            # Only use if select attributes are available and the product is not a duplicate in the dataset
            if entry['category'] and entry['description'] and entry['price'] and entry['title'] and entry['also_buy'] \
                    and entry['asin'] not in index_map:
                products.append(entry['asin'])
                co_purchase_list.append(entry['also_buy'])
                # Fill set with products that are connected in the graph
                linked_products.add(entry['asin'])
                for p in entry['also_buy']:
                    linked_products.add(p)
                # Creating hash map of products for easy lookup
                index_map[entry['asin']] = product_number
                # Calculating total number of edges to create edge matrix
                total_edges += len(entry['also_buy'])
                product_number += 1
        # adjacency_matrix = dok_matrix((len(products), len(products)))
        # Check if all connections are mutual
        total_non_mutual = 0
        with tqdm(total=len(products)) as pbar:
            for i, prod in enumerate(products):
                co_purchase = co_purchase_list[i]
                for co_p in co_purchase:
                    if co_p in index_map:
                        co_p_index = index_map[co_p]
                        if prod not in co_purchase_list[co_p_index]:
                            total_non_mutual += 1
            pbar.update()
        print("Percentage of non-mutual: {}".format(total_non_mutual/total_edges))
        # Remove products that are isolated in the graph, which means there is no connections to and from that book
        indices_to_remove = []
        # print("Finding indexes of isolated books.")
        # with tqdm(total=len(products)) as pbar:
        #     for asin, index in index_map.items():
        #         if asin not in linked_products:
        #             indices_to_remove.append(index)
        #         pbar.update()

        # print("Removing {} isolated books from the graph.".format(len(indices_to_remove)))
        # with tqdm(total=len(indices_to_remove)) as pbar:
        #     for index in sorted(indices_to_remove, reverse=True):
        #         products.pop(index)
        #         co_purchase_list.pop(index)
        #         pbar.update()
        #
        # index_map_new = {key: val for key, val in tqdm(index_map.items()) if val not in indices_to_remove}
        # print("New Number of Products:")
        # print(len(index_map_new))

        print("Number CoPurchase Lists:")
        print(len(co_purchase_list))

        print("Number of products: {}".format(len(index_map)))

        # Connectivity matrix, which has two entries for each of the nodes connected to the edge like
        # [1, 0, 3], [2, 3, 2] denotes edges 1 to 2, 0 to 3 and 3 to 2.
        graph_connectivity = np.zeros(shape=(2, total_edges))
        edge_index = 0

        # Fill the connectivity matrix with book co-purchase data
        product_number_graph = 0
        with tqdm(total=len(products)) as pbar:
            for j, co_purchase in enumerate(co_purchase_list):
                for product in co_purchase:
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
    return graph_connectivity, index_map


def node2vec(connectivity):
    edges = []
    for i in range(connectivity.shape[1]):
        edges.append((connectivity[0][i], connectivity[1][i]))
    edges = np.array(edges)

    graph = Graph(edges,
                  directed=False, weighted=False)
    n2v = Node2Vec(graph, dim=300, walk_length=100, context=10, p=2.0, q=0.5, workers=14)
    n2v.train(epochs=20)
    print(n2v.wv)


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data'
    connectivity_array, index_map_output = load_graph(data_path)
    # print(set(connectivity_array.flatten()))
    np.save("data/connectivity_array", connectivity_array)
    with open("data/index_mapping.pkl", 'wb') as f:
        pickle.dump(index_map_output, f)
    graph_matrix = torch.tensor(connectivity_array)
    # node2vec(connectivity_array)
    # data = Data(edge_index=graph_matrix)
    # torch.save(data, "../data/graph_object_no_features")
