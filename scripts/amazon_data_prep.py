import os
import json
import gzip
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import regex as re
from pathlib import Path
import nltk
from nltk.corpus import stopwords
# Comment this out if already downloaded
nltk.download("stopwords")


# load the meta data
data_path = Path(__file__).parent.parent / 'data'
data = []
with gzip.open(data_path / 'meta_Books.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))
    
# convert list into pandas dataframe
df = pd.DataFrame.from_dict(data)
del data

# remove rows with unformatted title (i.e. some 'title' may still contain html style content)
df = df.fillna('')
df = df[~df.title.str.contains('getTime')] # filter those unformatted rows

# how those unformatted rows look like
df.iloc[118]

# keep only books which have at least one co-purchasing link
df = df.loc[df['also_buy'].apply(len) > 0]

# keep only vars of interest
vars_to_keep = ['asin', 'rank', 'title', 'description', 
                'price', 'category', 'also_buy']
df = df.loc[:, vars_to_keep]

# keep only items for which we observe variables of interest
df = df.loc[df['title'] != '']
df = df.loc[df['description'].apply(len) > 0]
df = df.loc[df['price'] != '']
df = df.loc[df['category'].apply(len) > 0]
df.reset_index(inplace=True, drop = True)
len(df)

# clean categories
for i in range(len(df)):
    df.loc[i, "category"] = pd.Series(df.loc[i, "category"]).str.replace("&amp;", "&").to_list()

# make category list to dummies
mlb = MultiLabelBinarizer()

cat_label_df = pd.DataFrame(mlb.fit_transform(df['category']), columns = mlb.classes_, index = df.index)

# remove columns which include numbers
indices_to_remove = []
for i in range(len(cat_label_df.columns)):
    if any(char.isdigit() for char in cat_label_df.columns[i]):
        indices_to_remove.append(i)
cat_label_df = cat_label_df.drop(labels = cat_label_df.columns[indices_to_remove], axis = 1)

# also remove books column as common to all items
cat_label_df.drop(labels = ['Books'], axis = 1, inplace=True)
df = pd.concat([df, cat_label_df], axis = 1)
del cat_label_df
df.drop(labels = ['category'], axis = 1, inplace=True)

# remove items with errors in parsed prices
df = df.loc[~df['price'].str.contains('action')]
df = df.loc[~df['price'].str.contains('script')]
df.reset_index(inplace=True, drop = True)

# clean price
df.loc[:, 'price'] = df['price'].str.replace(',', '').str.replace('$', '').astype(float)

# clean rank
df['rank'] = df['rank'].str.replace(',', '').str.extract(r'(\d*)')


def cleaner(document):
    
    # correct html parsing errors
    doc = document.replace("&amp;", "and")
    doc = doc.replace("&amp;", "and")
    doc = doc.replace("&quot;", '"')
    doc = doc.replace("&#39;", "'")
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    doc = re.sub(cleanr, ' ', doc)
    
    # remove non ascii chars
    doc = re.sub(r'[^\x00-\x7F]+',' ', doc)
    
    # remove decimal points and spaces in numbers
    doc = re.sub(r'(?<=\d)[,\.\s\-](?=\d)', '' , doc)

    # remove numbers
    doc = re.sub(r' \d+\s?%?(?=[\s\.])', ' ' , doc)
    doc = re.sub(r'[<>]?\d+(?=[\s%,;:–\.])', ' ', doc)

    # remove line breaks
    doc = re.sub(r'\n', " ", doc)

    # replace :, ?, ! by ., also several consecutive occurrences
    doc = re.sub(r'[:\?!]+', ".", doc)

    # remove hyphens only when not connecting two words (minus signs)
    doc = re.sub(r'-(?![a-zA-Z])|(?<![a-zA-Z])-', '', doc)

    # remove quotation marks for quotes
    doc = re.sub(r'‘(.*?)’', r'\1', doc)
    doc = re.sub(r'“(.*?)”', r'\1', doc)
    doc = re.sub(r'"(.*?)"', r'\1', doc)

    # keep only chars, dots and spaces
    doc = re.sub(r'[^a-zA-Z\.\s\-]', ' ', doc)
    
    # remove punctuation
    # doc = re.sub(r'[^\P{P}-]+', ' ', doc)

    # remove double spaces
    doc = re.sub(r'\s{2,}', " ", doc)

    # remove trailing spaces
    doc = doc.strip()

    return doc

# clean title
df['clean_title'] = df['title'].apply(cleaner)
# remove single letter words
docs = [[w.lower() for w in doc.split()] for doc in df['clean_title']]
docs = [[w for w in docs[doc] if len(w) > 1] for doc in range(len(docs))]
df['clean_title'] = [" ".join(doc) for doc in docs]


# clean descriptions
# concatenate description items
docs = [" ".join(doc) for doc in df['description']] 
df['clean_description'] = pd.Series(docs).apply(cleaner)

# Remove stopwords (word2vec doesn't want them so neither do I!)
stop = stopwords.words('english')
df['clean_description'] = df['clean_description'].apply(lambda x: ' '.join([word for word in x.split()
                                                                           if word not in stop]))
print(stop)
print(df['clean_title'])
df['clean_title'] = df['clean_title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
print(df['clean_title'])
# remove single letter words
docs = [[w.lower() for w in doc.split()] for doc in df['clean_description']]
docs = [[w for w in docs[doc] if len(w) > 1] for doc in range(len(docs))]
df['clean_description'] = [" ".join(doc) for doc in docs]
del docs

# remove duplicates asin
df.drop_duplicates(subset = "asin", keep = 'first', inplace=True)
df.reset_index(inplace=True, drop = True)

# remove books which have an empty title / description now
df = df.loc[df['clean_title'] != '']
df = df.loc[df['clean_description'] != '']

# remove categories with only 1 book in them
cat_sum = df.iloc[:, 6:669].sum(axis = 0)
rem_cats = cat_sum[cat_sum == 1].index.to_list()
df.drop(labels=rem_cats, axis = 1, inplace=True)
del rem_cats

# remove books that are not assigned to any category
book_cat_sum = df.iloc[:, 6: 509].sum(axis = 1)
df = df.loc[(book_cat_sum > 0).index]

df.reset_index(inplace=True, drop = True)
df.to_parquet(data_path / 'cleaned_data.gzip', compression = 'gzip')




# need validation AND test set

# test with some manually selected books that were published after dataset got scraped!