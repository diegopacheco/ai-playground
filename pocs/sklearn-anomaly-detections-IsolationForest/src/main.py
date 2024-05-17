import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import IsolationForest

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # use only the first two features

# Fit the model
clf = IsolationForest(max_samples=100, random_state=42)
clf.fit(X)

# Predict the anomalies in the data
y_pred = clf.predict(X)

# Plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(4, 8, 500), np.linspace(1.5, 4.5, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X[:, 0], X[:, 1], c='white',
                 s=20, edgecolor='k')

plt.axis('tight')
plt.xlim((4, 8))
plt.ylim((1.5, 4.5))
plt.show()