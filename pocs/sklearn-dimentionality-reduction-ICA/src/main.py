from sklearn.decomposition import FastICA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply ICA
ica = FastICA(n_components=2)  # We reduce to 2 dimensions
X_ica = ica.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_ica[y == 0, 0], X_ica[y == 0, 1], color='red', alpha=0.5,label='Iris-setosa')
plt.scatter(X_ica[y == 1, 0], X_ica[y == 1, 1], color='blue', alpha=0.5,label='Iris-versicolor')
plt.scatter(X_ica[y == 2, 0], X_ica[y == 2, 1], color='green', alpha=0.5,label='Iris-virginica')
plt.legend()
plt.title('ICA of IRIS dataset')
plt.xlabel('IC1')
plt.ylabel('IC2')
plt.show()