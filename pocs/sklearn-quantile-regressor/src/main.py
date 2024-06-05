import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor

# Load California housing dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create QuantileRegressor object
qr = QuantileRegressor(quantile=0.5, alpha=0.01)

# Train the model using the training sets
qr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = qr.predict(X_test)

# Plot outputs
plt.scatter(y_test, y_pred, color='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Quantile Regression')
plt.show()