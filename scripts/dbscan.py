import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN


data_path = Path(__file__).parent.parent / 'data'
df = pd.read_parquet(data_path / 'embed_data.gzip')

testembed = [row[None,:] for row in df["title_embed"]]
testembed = np.concatenate(testembed)
testembed.shape

# run DBSCAN
clustering = DBSCAN(eps=3, min_samples=10, n_jobs = 1).fit(testembed)
clustering.labels_

# 12:18 start

# which metric to use? 
# eucliean with high dimeions leads to curse of dimensionality -> only cluster based on node embeddings!
# need for efficient solution
# min_samples = 10 at least




################################################################################
# KMEANS    

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=20, random_state=0).fit(testembed)
label_list = kmeans.labels_

u, counts = np.unique(label_list, return_counts=True)

test = [i for i in range(len(label_list)) if label_list[i] == 2]
# 1 = cookbooks
# 0 = animals/comics/childrens
# geographical guides, etc.
df.loc[test, "clean_title"]

# check if books in same cluster have similar titles?


# check title of 544490, Wall Calendar 2019 [12 pages 8&quot;x11&quot;]
