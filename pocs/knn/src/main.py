import numpy as np
from collections import Counter

def kNN(X, X_training, y_training, k=3):
    """
    Returns the kNN prediction of the data sample X, based on the training data
    (X_training, y_training), and the k value.

    Parameters:
    -----------
    X : numpy.ndarray
        The new data sample of shape (n_features, ).

    X_training : numpy.ndarray
        The training data samples of shape (n_samples, n_features).

    y_training : numpy.ndarray
        Classification labels for the training samples of shape (n_samples, 1).

    k : int, optional
        The number of nearest neighbors to check. Default is 3.

    Returns:
    --------
    int
        The predicted class label.
    """

    # 1. Calculating the distances of X from the instances in X_training.
    distances = np.linalg.norm(X_training - X, axis=1)

    # 2. Finding the k nearest neighbors.
    k_nearest_idx = np.argsort(distances)[:k]

    # 3. Taking a vote among the k nearest neighbors.
    k_nearest_labels = y_training[k_nearest_idx]                        # the labels of the k nearest neighbors
    label_counts = Counter(k_nearest_labels)                            # counting the labels
    prediction, _ = max(label_counts.items(), key=lambda x: x[1])       # finding the label with the highest count

    return prediction

X = np.random.rand(3)
X_training = np.random.rand(10, 3)
y_training = np.random.randint(0, 2, size=30)
print(f"X training data: {X_training} Y training data: {y_training} - X is {X}")

prediction = kNN(X, X_training, y_training)
print(f"Prediction: {prediction}")