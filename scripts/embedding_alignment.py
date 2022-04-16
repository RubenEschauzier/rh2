from node2vec import *
from word2vec import *
import torch as t
from torch.utils.data import Dataset


class EmbeddingAlignment(t.nn.Module):

    def __init__(self, graph, gamma, b):
        super().__init__()
        self.graph = graph
        self.gamma = gamma
        self.b = b
        # Cosine similarity across the first dimension
        self.cosine_similarity = t.nn.CosineSimilarity(dim=1)

    def rank_loss(self, batch, links, gamma, beta):
        """

        :param batch: Batch of node pairs (B), with shape [num_pairs, 2, embedding_size)
        :param links: Batch of variable indicating if node pairs are neighbours (y_i), with shape [num_pairs, 1)
        :param gamma: Hyper parameter for logistic loss
        :param beta: Hyper parameter for logistic loss
        :return: The rank_loss for one batch (might want to do all batches at once?)
        """
        num_pairs = batch.shape[0]
        neg_part = (1-links)*self.logistic_loss(self.cosine_similarity(batch).neg())
        pos_part = links*self.logistic_loss(self.cosine_similarity(batch))
        return t.sum((neg_part + pos_part))/num_pairs

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


class NodePairBatchLoader(Dataset):

    def __init__(self):
        super().__init__()
        pass
