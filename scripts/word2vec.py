import pickle

import gensim.downloader as api
import nltk.tokenize
import pandas as pd
from nltk import ngrams, FreqDist
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


class Word2VecPy(torch.nn.Module):

    def __init__(self, negative_sampling, num_neg, window_size, gensim_model, vec_dim):
        super(Word2VecPy, self).__init__()
        self.freq_dict = {}
        self.vocab = {}
        self.vec_dim = vec_dim
        self.neg_s = negative_sampling
        self.num_neg = num_neg
        self.window_size = window_size
        self.gensim_model = api.load(gensim_model)
        self.weights = None
        self.embedding_in = torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.gensim_model.vectors))
        self.embedding_out = torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.gensim_model.vectors))

        output_dim = self.neg_s + 1
        # Here we use log sigmoid, not sure if correct, should check with paper
        self.output_layer = torch.nn.Linear(self.vec_dim, output_dim)

    def build_vocab(self, corpus, save, path=None, tokenized=None):
        """
        Builds vocabulary and frequency dicts out of a corpus to use as model input
        :param corpus: Array-like with un-tokenized sentences as list elements.
        :param save: If we want to save the output to a file
        :param path: Path to save output to
        :param tokenized: Optional parameter to supply the tokenized corpus, if not supplied will create tokenized corpus

        :return: void
        """
        tokenized_corpus = []
        if not tokenized:
            for sentence in tqdm(corpus):
                tokenized_corpus.append(nltk.tokenize.word_tokenize(sentence))
        else:
            tokenized_corpus = tokenized

        flat_tokenized_corpus = [item for sublist in tokenized_corpus for item in sublist]
        self.freq_dict = FreqDist(flat_tokenized_corpus)
        self.vocab = self.gensim_model.key_to_index
        if save and path:
            with open(path / 'frequency_dict.pkl', 'wb') as f:
                pickle.dump(self.freq_dict, f)
            with open(path / 'vocab.pkl', 'wb') as f:
                pickle.dump(self.vocab, f)
        elif save and not path:
            raise ValueError("Provide path to save to")

    def build_frequency_array(self):
        freq_array = np.zeros(len(self.vocab))
        if self.freq_dict and self.vocab:
            for key, value in self.vocab.items():
                if self.freq_dict[key] > 0:
                    freq_array[value] = self.freq_dict[key]
        return freq_array

    def build_weights(self, weights):
        """
        Function to determine sampling weights of words in corpus
        :param weights: Frequencies of words in corpus
        :return: void
        """
        self.weights = np.power(weights, 0.75)
        self.weights = self.weights / self.weights.sum()
        self.weights = torch.FloatTensor(self.weights)
        print(self.weights)

    def load_vocab_freq(self, path_vocab, path_freq, encoding='pickle'):
        """
        Function to load vocab, since building it each time can take long
        :param path_vocab: Path where vocab is stored
        :param path_freq: Path where frequency dict is stored
        :param encoding: Type of parameter (valid: 'pickle')

        :return: void
        """
        with open(path_vocab, 'rb') as f:
            if encoding == 'pickle':
                self.vocab = pickle.load(f)
            else:
                raise ValueError("Not a valid encoding")
        with open(path_freq, 'rb') as f:
            if encoding == 'pickle':
                self.freq_dict = pickle.load(f)

    def forward_tgt(self, input_batch):
        input_vec = self.embedding_in(input_batch)
        return input_vec

    def forward_ctx(self, output_batch):
        output_vec = self.embedding_out(output_batch)
        return output_vec

    def loss(self, target, context, negative_samples):
        batch_size = target.shape[0]
        tgt_emb = self.forward_tgt(target)
        ctx_emb = self.forward_ctx(context)
        neg_emb = self.forward_ctx(negative_samples).neg()

        pos_loss = torch.bmm(ctx_emb, tgt_emb).squeeze().sigmoid().log().mean(1)
        neg_loss = torch.bmm(neg_emb, tgt_emb).squeeze().sigmoid().log().view(-1, context.shape[1], self.num_neg)
        neg_loss = neg_loss.sum(2).mean(1)

        loss = -(pos_loss+neg_loss).mean()

        return loss


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data'
    cleaned_data = pd.read_parquet(data_path / 'cleaned_data.gzip')
    use_neg_sampling = True
    model_to_use = 'word2vec-google-news-300'
    vec_dim = 300

    # word2vec_model = Word2VecPy(use_neg_sampling, 10, 2, model_to_use, vec_dim)
    # word2vec_model.build_vocab(cleaned_data['clean_description'], True, data_path)
    #
    word2vec_model = Word2VecPy(use_neg_sampling, 10, 2, model_to_use, vec_dim)
    word2vec_model.load_vocab_freq(data_path / 'vocab.pkl', data_path / 'frequency_dict.pkl')
    freq_arr = word2vec_model.build_frequency_array()
    word2vec_model.build_weights(freq_arr)
