import gzip
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import scipy

from scipy.sparse import dok_matrix, coo_matrix
from tqdm import tqdm
import torch
from fastnode2vec import Graph, Node2Vec

print(scipy.version.version)


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
        products = [];
        co_purchase_list = [];
        index_map = {}
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
        print("Percentage of non-mutual: {}".format(total_non_mutual / total_edges))

        for i, purchases in enumerate(co_purchase_list):
            for product in purchases:
                pass

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
        with tqdm(total=len(products)) as pbar:
            for j, co_purchase in enumerate(co_purchase_list):
                for product in co_purchase:
                    if product in index_map:
                        graph_connectivity[0][edge_index] = j
                        graph_connectivity[1][edge_index] = index_map[product]
                        edge_index += 1
                pbar.update()
        graph_connectivity = graph_connectivity[:, :edge_index]
        print(len(set(graph_connectivity.flatten())))
        # for i in range(len(products)):
        #     if adjacency_matrix[i,:].count_nonzero() == 0 and adjacency_matrix[:,i].count_nonzero() == 0:
        #
        #
        # print(adjacency_matrix.count_nonzero())
    return graph_connectivity, index_map


def load_graph_new(path):
    node_attr = pd.read_parquet(path / 'embed_data.gzip')
    filtered_co_purchase = []
    index_map = {}
    filtered_index_map = {}
    co_purchase_map = {}
    filtered_purchase_map = {}
    index_co_purchase_map = {}
    products_to_remove = {}
    total_edges = 0

    # Create mapping of all books in cleaned data
    for idx, asin in enumerate(node_attr['asin']):
        index_map[asin] = idx

    co_purchase_index = 0
    for products in node_attr['also_buy']:
        for product in products:
            if product in index_map:
                co_purchase_map[product] = co_purchase_index
                co_purchase_index += 1

    # Create mapping of all co-purchased books that are in the cleaned data

    # start_index_co_purchase = max(index_map.values())
    # for co_purchase in node_attr['also_buy']:
    #     for product in co_purchase:
    #         if product not in index_map:
    #             index_map[product] = start_index_co_purchase
    #             start_index_co_purchase += 1
    filtered_co_purchase_idx = 0
    for i, (asin, co_purchase) in enumerate(zip(node_attr['asin'], node_attr['also_buy'])):
        # Remove all co-purchase links if it is not in our original cleaned data
        new_co_purchase = [product for product in co_purchase if product in index_map]
        # If the product has a co purchase link then add it to the new filtered copurchase data
        if new_co_purchase:
            total_edges += len(new_co_purchase)
            filtered_co_purchase.append(new_co_purchase)
            index_co_purchase_map[asin] = new_co_purchase
            # Make new map with all products in the filtered version of the co-purchases
            for product_id in new_co_purchase:
                filtered_purchase_map[product_id] = filtered_co_purchase_idx
                filtered_co_purchase_idx += 1

    for i, (asin, co_purchase) in enumerate(zip(node_attr['asin'], node_attr['also_buy'])):
        if asin not in filtered_purchase_map and asin not in index_co_purchase_map:
            index_map.pop(asin)
            products_to_remove[asin] = i

    new_start_index = 0
    for asin in node_attr['asin']:
        if asin not in products_to_remove:
            filtered_index_map[asin] = new_start_index
            new_start_index += 1

    graph_connectivity = np.zeros(shape=(2, total_edges))
    edge_index = 0

    for i, (asin, co_purchase) in enumerate(index_co_purchase_map.items()):
        for product in co_purchase:
            graph_connectivity[0][edge_index] = filtered_index_map[asin]
            graph_connectivity[1][edge_index] = filtered_index_map[product]
            edge_index += 1
    graph_connectivity = graph_connectivity[:, :edge_index]

    num_nodes = len(set(graph_connectivity.flatten()))
    sparse_adj_matrix = dok_matrix((num_nodes, num_nodes), dtype=int)

    with tqdm(total=graph_connectivity.shape[0]) as pbar:
        for start, end in zip(graph_connectivity[0], graph_connectivity[1]):
            sparse_adj_matrix[start, end] = 1
            pbar.update()

    return graph_connectivity, filtered_index_map, sparse_adj_matrix


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
    connectivity_array, index_map_output, adj_matrix = load_graph_new(data_path)
    # connectivity_array, index_map_output = load_graph(data_path)
    # print(set(connectivity_array.flatten()))
    # np.save("data/connectivity_array", connectivity_array)
    # with open("data/index_mapp ing.pkl", 'wb') as f:
    #     pickle.dump(index_map_output, f)
    with open("data/sparse_adj_matrix.pkl", 'wb') as f_s:
        pickle.dump(adj_matrix, f_s)
    # graph_matrix = torch.tensor(connectivity_array)
    # node2vec(connectivity_array)
    # data = Data(edge_index=graph_matrix)
    # torch.save(data, "../data/graph_object_no_features")
