from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply NMF
nmf = NMF(n_components=2, init='random', random_state=0)  # We reduce to 2 dimensions
X_nmf = nmf.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_nmf[y == 0, 0], X_nmf[y == 0, 1], color='red', alpha=0.5,label='Iris-setosa')
plt.scatter(X_nmf[y == 1, 0], X_nmf[y == 1, 1], color='blue', alpha=0.5,label='Iris-versicolor')
plt.scatter(X_nmf[y == 2, 0], X_nmf[y == 2, 1], color='green', alpha=0.5,label='Iris-virginica')
plt.legend()
plt.title('NMF of IRIS dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()