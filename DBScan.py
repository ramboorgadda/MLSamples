from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
X,y= make_moons(n_samples=250, noise=0.05, random_state=42)
plt.scatter(X[:,0],X[:,1],c=y)
plt.title('Synthetic Data Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f'Scaled Features:\n{X_scaled[:5]}')
# DBSCAN Clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X_scaled)
labels = dbscan.labels_
print(f'Labels: {labels}')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()