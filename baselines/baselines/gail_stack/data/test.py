import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture as GMM

data = np.load('stack_demo.npz')
obs = data['obs']
obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])[:1000]

#bandwidth = estimate_bandwidth(obs, quantile=0.2, n_samples=500)
#model = MeanShift(band1width=bandwidth, bin_seeding=True)
gmm = GMM(n_components=10)
model = gmm.fit(obs)
y = model.predict(obs)


labeled_x = [[] for i in range(len(y))]
for i in labeled_x:
	if len(i) == 0:
		del i


for i in range(len(obs)):
   labeled_x[y[i]].append(obs[i])
labeled_x = np.array(labeled_x)

import pandas as pd


feat_cols = [ 'pixel'+str(i) for i in range(obs.shape[1]) ]

df = pd.DataFrame(obs, columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

rndperm = np.random.permutation(df.shape[0])


import time

from sklearn.manifold import TSNE

n_sne = 7000

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

from ggplot import *

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by digit")

print(chart)
