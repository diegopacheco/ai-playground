from sklearn.decomposition import KernelPCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#
#  Generalized Discriminant Analysis (GDA) 
#  is not directly available as a method in the sklearn library.
#  GDA is a non-linear version of Linear Discriminant 
#  Analysis (LDA), where kernel functions are used to 
#  map input vectors into high-dimensional feature space.
#  Kernel PCA, which is a similar concept.

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf')  # We reduce to 2 dimensions
X_kpca = kpca.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', alpha=0.5,label='Iris-setosa')
plt.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', alpha=0.5,label='Iris-versicolor')
plt.scatter(X_kpca[y == 2, 0], X_kpca[y == 2, 1], color='green', alpha=0.5,label='Iris-virginica')
plt.legend()
plt.title('Kernel PCA of IRIS dataset')
plt.xlabel('Kernel PC1')
plt.ylabel('Kernel PC2')
plt.show()