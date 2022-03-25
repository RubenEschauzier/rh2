import pandas as pd
import numpy as np
from pathlib import Path
import spacy
from tqdm import tqdm
# download language model
# !python -m spacy download en_core_web_lg
# nlp = spacy.load('en_core_web_lg', disable=["tagger", "attribute_ruler", "lemmatizer"])

nlp = spacy.load('en_core_web_lg')


# remove stop words ?

# load the meta data
data_path = Path(__file__).parent.parent / 'data'
df = pd.read_parquet(data_path / 'cleaned_data.gzip')

# process titles, get mean vector for each title (300 dimensional)
raw_titles = df['clean_title'].to_list()
title_embeds = [nlp(raw_title).vector for raw_title in tqdm(raw_titles)]
df['title_embed'] = title_embeds
del title_embeds, raw_titles



# process descriptions, get mean vector for each description (300 dimensional)
raw_descs = df['clean_description'].to_list()
desc_embeds = [nlp(raw_desc).vector for raw_desc in tqdm(raw_descs)]
df['description_embed'] = desc_embeds
del desc_embeds, raw_descs




df.drop(labels = ['clean_title', 'clean_description'], axis = 1, inplace = True)
df.to_parquet(data_path / 'embed_data.gzip', compression = 'gzip')




# check if vectors of books in a small subcategory are similar



