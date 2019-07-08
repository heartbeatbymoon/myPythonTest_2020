# encoding=utf-8
# 认真总结，可以作为聚类算法的模板
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

beer = pd.read_csv("G:\\datas\\ai\\data.txt", sep=" ")
print(beer.head(5))

X = beer[["calories", "sodium", "alcohol", "cost"]]

km = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=2).fit(X)

# 查看结果array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 0, 2, 1])
# 第一行数据的结果为哪个类别第0组，。。。。。
print(km.labels_)
beer['cluster'] = km.labels_
beer["cluster2"] = km2.labels_

# 排序的目的是啥？
print(beer.sort_values("cluster"))

# from pandas.tools.plotting import scatter_matrix
cluster_centers = km.cluster_centers_
cluster_centers_2 = km2.cluster_centers_

# 查看每个堆的平均值
print(beer.groupby("cluster").mean())
print(beer.groupby("cluster2").mean())

centers = beer.groupby("cluster").mean().reset_index()

print(centers)

# plt.rcParams["font.size"] = 14
# colors = np.array(["red", "green", "blue", "yellow"])

# plt.scatter(beer["calories"], beer["alcohol"], c=colors[beer["cluster"]])
#
# plt.xlabel("Calories")
# plt.ylabel("Alcohol")
# plt.show()

# pd.scatter_matrix(beer[["calories", "sodium", "alcohol", "cost"]], s=100, alpha=1, c=colors[beer["cluster"]],
#                   figsize=(10, 10))
# plt.suptitle("With 3 centroids initialized")

# plt.show()

# 不一定做归一化一定好，因为可能有些数据真的很重要
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
# print(X_scaler)

km = KMeans(n_clusters=3).fit(X_scaler)
beer["scaled_cluster"] = km.labels_

# 为什么要排序》？？？？
beer.sort_values("scaled_cluster")
print(beer.groupby("scaled_cluster").mean())

# pd.scatter_matrix(X, c=colors[beer.scaled_cluster], alpha=1, figsize=(10, 10), s=100)
# plt.show()


from sklearn import metrics
# score_scaled = metrics.silhouette_score(X,beer.scaled_cluster)
# score = metrics.silhouette_score(X,beer.cluster)

scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X,labels)
    scores.append(score)
print(scores)

plt.plot(list(range(2,20)),scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")
plt.show()