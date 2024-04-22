import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt

# Generate fake data
np.random.seed(0)
data = np.r_[np.random.randn(50, 2) + [2, 2], np.random.randn(50, 2) + [4, 4]]

# KMeans clustering
centroids, _ = kmeans(data, 2)

# Assign clusters
clusters, _ = vq(data, centroids)

# Plot results
plt.scatter(data[:, 0], data[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r')
plt.show()