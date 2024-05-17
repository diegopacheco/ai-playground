from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca = PCA(n_components=2)  # We reduce to 2 dimensions
X_pca = pca.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='red', alpha=0.5,label='Iris-setosa')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='blue', alpha=0.5,label='Iris-versicolor')
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], color='green', alpha=0.5,label='Iris-virginica')
plt.legend()
plt.title('PCA of IRIS dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()