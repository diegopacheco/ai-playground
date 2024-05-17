import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from scipy import stats

# Load the California Housing dataset
california = fetch_california_housing()
x = california.data

# Calculate the z-score of each value in the dataset, this gives us a measure of how many standard deviations an element is from the mean
z_scores = stats.zscore(x)

# Define a threshold to identify an outlier
threshold = 3

# Get the positions of the outliers in all the columns of the dataset
outliers = np.where(np.abs(z_scores) > threshold)

# Print the positions of the outliers
print(outliers)

# Plot the z-scores for each feature
for i in range(z_scores.shape[1]):
    plt.figure(figsize=(8,6))
    plt.scatter(range(z_scores.shape[0]), z_scores[:, i])
    plt.title(f'Z-scores of California Housing dataset - Feature {i+1}')
    plt.show()