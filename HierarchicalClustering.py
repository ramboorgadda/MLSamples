import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris=datasets.load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data.shape)
iris_data=pd.DataFrame(iris.data)
print(iris_data.head())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data)
print(iris_data_scaled[:5])
# Apply the PCA
from sklearn.decomposition import PCA
PCA_model = PCA(n_components=2)
iris_data_pca = PCA_model.fit_transform(iris_data_scaled)
print(iris_data_pca[:5])
plt.scatter(iris_data_pca[:, 0], iris_data_pca[:, 1], c=iris.target, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.colorbar(label='Species')
plt.show()
#Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(iris_data_pca, method='ward'))
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
agglo_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
agglo_model.fit(iris_data_pca)
print(f'Labels: {agglo_model.labels_}')
plt.scatter(iris_data_pca[:, 0], iris_data_pca[:, 1], c=agglo_model.labels_, cmap='viridis')
plt.show()
