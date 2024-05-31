import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Calculate the Manhattan distance between all pairs of points
distances = pdist(X, metric='cityblock')

# Convert the distances to a square matrix
dist_matrix = squareform(distances)

# Plot the distance matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(dist_matrix, cmap='viridis')
plt.title('Manhattan Distance Matrix')
plt.show()