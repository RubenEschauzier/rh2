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
        products = []; copurchase_list = []; index_map = {}
        linked_products = set()

        total_edges = 0
        product_number = 0
        print("Reading the data file.")
        for line in f:
            entry = json.loads(line.strip())
            if entry['category'] and entry['description'] and entry['price'] and entry['title']:
                products.append(entry['asin']); copurchase_list.append(entry['also_view'])
                # Fill set with products that are connected in the graph
                if entry['also_view']:
                    linked_products.add(entry['asin'])
                    for p in entry['also_view']:
                        linked_products.add(p)
                # Creating hash map of products for easy lookup
                index_map[entry['asin']] = product_number
                # Calculating total number of edges to create edge matrix
                total_edges += len(entry['also_view'])
                product_number += 1
        print(len(products))
        # adjacency_matrix = dok_matrix((len(products), len(products)))

        # Check if all connections are mutual
        total_non_mutual = 0
        with tqdm(total=len(products)) as pbar:
            for i, prod in enumerate(products):
                co_purchase = copurchase_list[i]
                for co_p in co_purchase:
                    if co_p in index_map:
                        co_p_index = index_map[co_p]
                        copurchase_of_product = copurchase_list[co_p_index]
                        if prod not in copurchase_list[co_p_index]:
                            print("Not Mutual!!")
                            total_non_mutual += 1
            pbar.update()
        print(total_non_mutual)
        # Remove products that are isolated in the graph
        indices_to_remove = []
        print("Finding indexes of isolated books.")
        with tqdm(total=len(products)) as pbar:
            for asin, index in index_map.items():
                if not copurchase_list[index] and asin not in linked_products:
                    indices_to_remove.append(index)
                if not copurchase_list and asin in linked_products:
                    print("Not mutual!")
                pbar.update()
        print("Removing the isolated books from the graph.")
        with tqdm(total=len(indices_to_remove)) as pbar:
            for index in sorted(indices_to_remove, reverse=True):
                products.pop(index)
                copurchase_list.pop(index)
                pbar.update()

        print(len(products))
        # Connectivity matrix, which has two entries for each of the nodes connected to the edge like
        # [1, 0, 3], [2, 3, 2] denotes edges 1 to 2, 0 to 3 and 3 to 2.
        graph_connectivity = np.zeros(shape=(2, total_edges))
        edge_index = 0

        with tqdm(total=len(products)) as pbar:
            for j, copurchase_list in enumerate(copurchase_list):
                for product in copurchase_list:
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
    #torch.save(data, "data/graph_object_no_features")