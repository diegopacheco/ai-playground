from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply LDA
lda = LDA(n_components=2)  # We reduce to 2 dimensions
X_lda = lda.fit_transform(X, y)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[y == 0, 0], X_lda[y == 0, 1], color='red', alpha=0.5,label='Iris-setosa')
plt.scatter(X_lda[y == 1, 0], X_lda[y == 1, 1], color='blue', alpha=0.5,label='Iris-versicolor')
plt.scatter(X_lda[y == 2, 0], X_lda[y == 2, 1], color='green', alpha=0.5,label='Iris-virginica')
plt.legend()
plt.title('LDA of IRIS dataset')
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.show()