from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)  # We reduce to 2 dimensions
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], color='red', alpha=0.5,label='Iris-setosa')
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], color='blue', alpha=0.5,label='Iris-versicolor')
plt.scatter(X_tsne[y == 2, 0], X_tsne[y == 2, 1], color='green', alpha=0.5,label='Iris-virginica')
plt.legend()
plt.title('t-SNE of IRIS dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()