from pathlib import Path
import numpy as np
from fastnode2vec import Graph, Node2Vec
import pickle


def node2vec(connectivity):
    edges = []
    for i in range(connectivity.shape[1]):
        edges.append((connectivity[0][i], connectivity[1][i]))
    edges = np.array(edges)

    graph = Graph(edges,
                  directed=True, weighted=False)
    n2v = Node2Vec(graph, dim=300, walk_length=80, context=10, p=2.0, q=1, batch_walks=10, workers=7)
    n2v.train(epochs=10)
    word_vectors = n2v.wv
    with open(data_path / 'node2vec_model.pkl', 'wb') as f:
        pickle.dump(n2v, f)


def test():
    with open(data_path / 'node2vec_model.pkl', "rb") as f_r:
        n2v = pickle.load(f_r)
    print(n2v.wv.get_vector(1))


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data'
    book_graph = np.load(str(data_path) + "\connectivity_array.npy")
    node2vec(book_graph)
    test()
