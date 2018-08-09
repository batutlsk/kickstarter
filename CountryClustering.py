# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:02:08 2017

@author: Sami
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:57:39 2017

@author: Sami
"""

from pandas import read_csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering


#Read the data
series = read_csv('cluster.csv', header=0, index_col=0)


X=series.as_matrix(columns=series.columns.tolist()[0:-1])

#X = series.values


#K-means Clustering
k_means = KMeans(init='k-means++', n_clusters=3, n_init=5)

k_means.fit(X)

labels = k_means.labels_
score = metrics.silhouette_score(X, labels, metric='euclidean')

#Hierarical Clustering

#'ward', 'average', 'complete'
clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)

clustering.fit(X)

labels2 = clustering.labels_
score2 = metrics.silhouette_score(X, labels2, metric='euclidean')