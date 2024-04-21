###PCA

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, cut_tree
from ISLP.cluster import compute_linkage


# us_arrests = get_rdataset('USArrests').data
# #print(us_arrests)
# # print(us_arrests.columns)
# # print(us_arrests.mean())
# #print(us_arrests.var())
# scaler = StandardScaler(with_std=True, with_mean=True)
# us_scaled = scaler.fit_transform(us_arrests)
# pca_us = PCA()
# pca_us.fit(us_scaled)
# #print(pca_us.mean_)
# scores = pca_us.transform(us_scaled)
# #print(scores)
# #print(pca_us.components_)
# i, j = 0, 1
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.scatter(scores[:, 0], scores[:, 1])
# ax.set_xlabel('PC%d' % (i+1))
# ax.set_ylabel('PC%d' % (j+1))
# for k in range(pca_us.components_.shape[1]):
#     ax.arrow(0, 0, pca_us.components_[i, k], pca_us.components_[j, k])
#     ax.text(pca_us.components_[i, k],
#             pca_us.components_[j, k],
#             us_arrests.columns[k])
# scale_arrow = s_ = 2
# scores[:, 1] *= -1
# pca_us.components_[1] *= -1
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.scatter(scores[:, 0], scores[:, 1])
# ax.set_xlabel('PC%d' % (i+1))
# ax.set_ylabel('PC%d' % (j+1))
# for k in range(pca_us.components_.shape[1]):
#     ax.arrow(0, 0, s_*pca_us.components_[i, k], s_*pca_us.components_[j, k])
#     ax.text(s_*pca_us.components_[j, k],
#             s_*pca_us.components_[j,k],
#             us_arrests.columns[k])
# a = scores.std(0, ddof=1)
# #print(a)
# b = pca_us.explained_variance_ratio_
# #print(b)
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# ticks = np.arange(pca_us.n_components_)+1
# ax = axes[0]
# ax.plot(ticks, pca_us.explained_variance_ratio_, marker='o')
# ax.set_xlabel('principal component')
# ax.set_ylabel('proportion of variance explained')
# ax.set_ylim([0, 1])
# ax.set_xticks(ticks)
# ax = axes[1]
# ax.plot(ticks,
# pca_us.explained_variance_ratio_.cumsum(), marker='o')
# ax.set_xlabel('Principal Component')
# ax.set_ylabel('Cumulative Proportion of Variance Explained')
# ax.set_ylim([0, 1])
# ax.set_xticks(ticks)
# plt.show()
# c = np.array([1, 2, 8, -3])
# np.cumsum(c)
# print(c)


###K-MEANS

##The seed() function initializes the random number generator with a specified seed value. When you set a seed value, you ensure that the sequence of random numbers
##generated is reproducible. In other words, every time you run the code with the same seed, you'll get the same sequence of random numbers.
##Here, X is initialized as a NumPy array of shape (50, 2). It contains random numbers sampled from a standard normal distribution
##(mean = 0, standard deviation = 1).
# np.random.seed(0)
# x = np.random.standard_normal((50, 2))
# # print(x)
# x[:25, 0] += 3
# x[:25, 1] -= 4
# # print(x)
# kmeans = KMeans(n_clusters=2, random_state=2, n_init=20).fit(x)
# a = kmeans.labels_
# # print(a)
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.scatter(x[:, 0], x[:, 1], c=kmeans.labels_)
# ax.set_title("k-means clustering results with k=2");
# plt.show()
#
# kmeans = KMeans(n_clusters=3, random_state=3, n_init=20).fit(x)
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.scatter(x[:, 0], x[:, 1], c=kmeans.labels_)
# ax.set_title("k-means clustering results with k=3")
# plt.show()
#
# kmeans1 = KMeans(n_clusters=3, random_state=3, n_init=1).fit(x);
# kmeans20 = KMeans(n_clusters=3, random_state=3, n_init=20).fit(x);
# b = kmeans1.inertia_, kmeans20.inertia_
# print(b)




###Hierarchical clustering
np.random.seed(0)
x = np.random.standard_normal((50, 2))
# print(x)
x[:25, 0] += 3
x[:25, 1] -= 4
# print(x)
hclust = AgglomerativeClustering
hc_comp = hclust(distance_threshold=0, n_clusters=None, linkage='complete')
hc_comp.fit(x)
hc_av = hclust(distance_threshold=0, n_clusters=None, linkage='average');
hc_av.fit(x)
hc_sing = hclust(distance_threshold=0, n_clusters=None, linkage='single');
hc_sing.fit(x);
d = np.zeros((x.shape[0], x.shape[0]));
for i in range((x.shape[0])):
    x_ = np.multiply.outer(np.ones(x.shape[0]), x[i])
    d[i] = np.sqrt(np.sum((x-x_)**2, 1));
hc_sing_pre = hclust(distance_threshold=0, n_clusters=None, metric='precomputed', linkage='single')
hc_sing_pre.fit(d)
cargs = {'color_threshold':-np.inf, 'above_threshold_color' : 'black'}
linkage_comp = compute_linkage((hc_comp))
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp, ax=ax, **cargs);
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp, ax=ax, color_threshold=4, above_threshold_color='black');
plt.show()
c = cut_tree(linkage_comp, n_clusters=4).T
# print(c)
d = cut_tree(linkage_comp, height=5)
# print(d)
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
hc_comp_scale = hclust(distance_threshold=0, n_clusters=None, linkage='complete').fit(x_scale)
linkage_comp_scale = compute_linkage(hc_comp_scale)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp_scale, ax=ax, **cargs)
ax.set_title("hierarchical clustering with scaled features")
plt.show()
x = np.random.standard_normal((30, 3))
corD = 1 - np.corrcoef(x)
hc_cor = hclust(linkage='complete', distance_threshold=0, n_clusters=None, metric='precomputed')
hc_cor.fit(corD)
linkage_cor = compute_linkage(hc_cor)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_cor, ax=ax, **cargs)
ax.set_title("complete linkage with correlation-based dissimilarity");
plt.show()






