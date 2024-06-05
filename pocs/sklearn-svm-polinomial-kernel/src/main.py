import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features
y = iris.target

# Make it a binary classification problem by removing the third class
X, y = X[y != 2], y[y != 2]

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# SVC with polynomial kernel
svc = svm.SVC(kernel='poly', degree=3).fit(X, y)

# Plot the decision boundary
plt.subplots_adjust(wspace=0.4, hspace=0.4)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVC with polynomial (degree 3) kernel')

plt.show()