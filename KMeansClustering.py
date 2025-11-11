import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.cluster import KMeans
X,y= make_blobs(n_samples=1000, centers=3, n_features=2,cluster_std=0.60, random_state=0)
print(X)
print(y)
plt.scatter(X[:,0],X[:,1],c=y)
plt.title('Synthetic Data Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
#feature scaling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Elbow Method to find k value
wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_train_scaled)
    wcss.append(kmeans.inertia_)
print(wcss)
#plot elbow curve
plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.show()
kmeans=KMeans(n_clusters=3, init='k-means++')
X_trained_predicted=kmeans.fit_predict(X_train_scaled)
Y_Predictd=kmeans.fit_predict(X_test_scaled)
plt.scatter(X_test[:,0],X_test[:,1],c=Y_Predictd)
plt.title('KMeans Clustering on Test Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
#validating the k value kneelocator and silhouette score
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
kneedle = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
k_value = kneedle.elbow
print(f'Optimal number of clusters (k): {k_value}')
silhoutte_coeffificient = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_train_scaled)
    silhoutte_coeffificient.append(silhouette_score(X_train_scaled, kmeans.labels_))
plt.plot(range(2, 11), silhoutte_coeffificient)
plt.xticks(range(2, 11))
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient for Different k Values')
plt.show()