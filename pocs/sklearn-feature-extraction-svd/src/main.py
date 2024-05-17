import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import TruncatedSVD

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Perform SVD
svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X)

# Plot the results
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('viridis', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()