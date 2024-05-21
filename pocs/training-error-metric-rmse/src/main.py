import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 300)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 300)

# Reshape x to be a 2D array
x = x[:, np.newaxis]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Perform linear regression
model = LinearRegression()
model.fit(x_train, y_train)

# Predict y values
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Calculate the root mean squared error
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print the root mean squared errors
print(f"RMSE for training set: {rmse_train}")
print(f"RMSE for testing set: {rmse_test}")

# Plot the RMSE
plt.bar(['Training', 'Testing'], [rmse_train, rmse_test])
plt.title('Root Mean Square Error (RMSE)')
plt.ylabel('RMSE')
plt.show()