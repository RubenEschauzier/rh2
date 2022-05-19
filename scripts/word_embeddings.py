import pandas as pd
import numpy as np
from pathlib import Path
import spacy
from tqdm import tqdm
import gensim
import gensim.downloader as api
from torch import nn
import torch
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import pickle

# download language model
# !python -m spacy download en_core_web_sm

data_path = Path(__file__).parent.parent / 'data'
df = pd.read_parquet(data_path / 'cleaned_data.gzip')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

raw_titles = df['clean_title'].to_list()
raw_descs = df['clean_description'].to_list()

# load pre-trained model
model = api.load('word2vec-google-news-300')

# transform words in to IDs (pre-trained)
token_titles = [[model.key_to_index[token] for token in doc.split() if token in model.key_to_index] for doc in raw_titles]
token_descs = [[model.key_to_index[token] for token in doc.split() if token in model.key_to_index] for doc in raw_descs]

# get word2id and id2word
word2id = model.key_to_index
id2word = model.index_to_key

# get embedding
weights = torch.FloatTensor(model.vectors) # or model.wv directly
embedding = nn.Embedding.from_pretrained(weights)

##############################################################################
# proof that it works
# check that this gives the correct embedding
idx = model.key_to_index['banana']
vector = model.vectors[idx]

i = torch.LongTensor([word2id['banana']])
embedding(i).numpy() == vector
##############################################################################  
   
# train test split
# num_docs = df.shape[0]
# trSize = int(np.floor(0.85*num_docs))
# tsSize = int(np.floor(0.10*num_docs))
# vaSize = int(num_docs - trSize - tsSize)

# shuffle
df = df.sample(frac=1, random_state=2).reset_index(drop=True)
df.drop(labels = ['clean_title', 'clean_description'], axis = 1, inplace = True)

# split
train_set = df.sample(frac=0.8, random_state=22)
test_set = df.drop(train_set.index)

# save data
pickle.dump(train_set, open( "../data/train_set.p", "wb" ))
pickle.dump(test_set, open( "../data/test_set.p", "wb" ))
pickle.dump(word2id, open( "../data/word2id.p", "wb" ))
pickle.dump(id2word, open( "../data/id2word.p", "wb" ))




