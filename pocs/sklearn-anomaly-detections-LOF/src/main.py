import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import LocalOutlierFactor

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # use only the first two features

# Fit the model
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)

# Anomaly score
Z = clf.negative_outlier_factor_

# Create a mask for normal (inliers) and anomalies (outliers)
mask = y_pred != -1

# Plot the level sets of the decision function
plt.title("Local Outlier Factor (LOF)")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')

# Plot the inliers
plt.scatter(X[mask, 0], X[mask, 1], s=60, color='blue', edgecolor='k', label='Inliers')

# Plot the outliers
plt.scatter(X[~mask, 0], X[~mask, 1], s=60, color='red', edgecolor='k', label='Outliers')

plt.axis('tight')
plt.xlim((4, 8))
plt.ylim((1.5, 4.5))
plt.legend()
plt.show()