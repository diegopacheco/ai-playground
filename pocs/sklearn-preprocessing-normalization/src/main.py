import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

# Load the California housing dataset
california = datasets.fetch_california_housing()
X = california.data

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the data and transform the data
X_normalized = scaler.fit_transform(X)

# Plot the original data
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(X, bins=50)
plt.title('Original Data')

# Plot the normalized data
plt.subplot(1, 2, 2)
plt.hist(X_normalized, bins=50)
plt.title('Normalized Data')

plt.show()