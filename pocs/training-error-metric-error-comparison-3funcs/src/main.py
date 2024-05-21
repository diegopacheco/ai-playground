import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Calculate the mean squared error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Calculate the mean absolute error
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# Calculate the root mean squared error
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Print the errors
print(f"MSE for training set: {mse_train}, testing set: {mse_test}")
print(f"MAE for training set: {mae_train}, testing set: {mae_test}")
print(f"RMSE for training set: {rmse_train}, testing set: {rmse_test}")

# Plot the errors
labels = ['Training', 'Testing']
mse_values = [mse_train, mse_test]
mae_values = [mae_train, mae_test]
rmse_values = [rmse_train, rmse_test]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mse_values, width, label='MSE')
rects2 = ax.bar(x, mae_values, width, label='MAE')
rects3 = ax.bar(x + width, rmse_values, width, label='RMSE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error')
ax.set_title('Errors by dataset and error type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()