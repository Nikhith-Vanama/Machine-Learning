import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
data= { 'Feature1' : [ 1.1,1.3,5.7,6.1,2.4,8.5,3.7 ],'Feature2' : [ 2.5,2.8,8.9,9.2,3.1,9.6,4.2 ] }
data=pd.DataFrame(data)
features = data.select_dtypes(include=np.number)
kmeans = KMeans(n_clusters=3, random_state=42)  # Change n_clusters as needed
kmeans_labels = kmeans.fit_predict(features)
gmm = GaussianMixture(n_components=3, random_state=42)  # Change n_components as needed
gmm_labels = gmm.fit_predict(features)
kmeans_silhouette = silhouette_score(features, kmeans_labels)
gmm_silhouette = silhouette_score(features, gmm_labels)
print("Silhouette Score for K-Means:", kmeans_silhouette)
print("Silhouette Score for EM (GMM):", gmm_silhouette)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title("K-Means Clustering")
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])
plt.subplot(1, 2, 2)
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=gmm_labels, cmap='viridis', s=50)
plt.title("EM Clustering (GMM)")
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])
plt.tight_layout()
plt.show()