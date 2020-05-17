import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
clust_df=pd.read_csv("C:/Users/yogesh yadav/Downloads/Cust_Segmentation.csv")
df=clust_df.drop("Address",axis=1)
X=df.values[:,1:]
X=np.nan_to_num(X)
clus_data=StandardScaler().fit_transform(X)
#modeling
k_means=KMeans(init="k-means++",n_clusters=3,n_init=12)
k_means.fit(X)
labels=k_means.labels_
df["clus_kms"]=labels #add columns in dataset
df.groupby('clus_kms').mean()  # We can easily check the centroid values by averaging the features in each cluster.
area = np.pi * ( X[:, 1])**2  #circle area for clustering
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.show()



