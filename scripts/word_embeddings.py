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

with open('../data/index_mapping.pkl', 'rb') as f:
    index_map = pickle.load(f)
print(df['asin'].shape)
print(len(index_map))
# for key, value in index_map.items():
#     if key not in df['asin']:
#         print(key)
# load pre-trained model
model = api.load('word2vec-google-news-300')

# transform words in to IDs (pre-trained)
token_titles = [[model.key_to_index[token] for token in doc.split() if token in model.key_to_index] for doc in raw_titles]
token_descs = [[model.key_to_index[token] for token in doc.split() if token in model.key_to_index] for doc in raw_descs]

# padding to max length for titles
title_lengths = [len(doc) for doc in token_titles]
max_title_length = max(title_lengths)

desc_lengths = [len(doc) for doc in token_descs]
max_desc_length = int(np.percentile(desc_lengths, 95))

# remove tokens going above max_desc_length
token_descs = [[token for i,token in enumerate(doc) if i < max_desc_length] for doc in token_descs]

# add padding to titles and descriptions
padded_title = [doc.extend([0] * (max_title_length - len(doc))) for doc in token_titles if len(doc) < max_title_length]
padded_desc = [doc.extend([0] * (max_desc_length- len(doc))) for doc in token_descs if len(doc) < max_desc_length]

df['padded_title_ids'] = token_titles
df['padded_desc_ids'] = token_descs

# get word2id and id2word
word2id = model.key_to_index
id2word = model.index_to_key

# get embedding
weights = torch.FloatTensor(model.vectors) # or model.wv directly
embedding = nn.Embedding.from_pretrained(weights, padding_idx = 0)

# make sure 0 index layer is padding
with torch.no_grad():
    embedding.weight[0] = torch.zeros(300)

# proof that it works
i = torch.LongTensor([0,2,0])
embedding(i).numpy()

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
# df = df.sample(frac=1, random_state=2).reset_index(drop=True)
# df.drop(labels = ['clean_title', 'clean_description'], axis = 1, inplace = True)


# split
# train_set = df.sample(frac=0.8, random_state=22)
# test_set = df.drop(train_set.index)

# save data
# pickle.dump(train_set, open( "../data/train_set.p", "wb" ))
# pickle.dump(test_set, open( "../data/test_set.p", "wb" ))
df_to_save = df[['asin', 'padded_title_ids', 'padded_desc_ids']]
df_to_save.to_pickle('../data/padded_attr_df.pkl')
# pickle.dump(word2id, open( "../data/word2id.pkl", "wb" ))
# pickle.dump(id2word, open( "../data/id2word.pkl", "wb" ))




