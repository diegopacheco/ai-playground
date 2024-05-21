import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 300)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 300)

# Reshape x to be a 2D array
x = x[:, np.newaxis]

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)

train_error = []
val_error = []

# Calculate training and validation error for polynomial degrees from 1 to 10
for degree in range(1, 11):
    # Transform x to have polynomial features
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly_train = polynomial_features.fit_transform(x_train)
    x_poly_val = polynomial_features.transform(x_val)

    # Perform polynomial regression on the polynomial features
    model = LinearRegression()
    model.fit(x_poly_train, y_train)

    # Predict y values
    y_train_pred = model.predict(x_poly_train)
    y_val_pred = model.predict(x_poly_val)

    # Calculate the mean squared error
    train_error.append(mean_squared_error(y_train, y_train_pred))
    val_error.append(mean_squared_error(y_val, y_val_pred))

# Plot the training and validation errors
plt.plot(range(1, 11), train_error, color='blue', label='Training error')
plt.plot(range(1, 11), val_error, color='red', label='Validation error')
plt.title('Training and Validation error')
plt.xlabel('Degree of polynomial')
plt.ylabel('Mean squared error')
plt.legend()
plt.show()