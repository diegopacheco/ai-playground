import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

# Download historical data as dataframe
data = yf.download('AAPL', start='2020-01-01', end='2022-12-31')

# Use only closing prices
prices = data['Close']

# Prepare data for GaussianProcessRegressor
X = np.linspace(0, len(prices)-1, len(prices)).reshape(-1, 1)
y = prices.values

# Create GaussianProcessRegressor object with RBF kernel
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel)

# Train the model
gpr.fit(X, y)

# Make predictions
X_pred = np.linspace(0, len(prices)-1, len(prices)).reshape(-1, 1)
y_pred, sigma = gpr.predict(X_pred, return_std=True)

# Plot outputs
plt.figure(figsize=(10, 5))
plt.plot(X, y, 'r:', label='AAPL Stock Price')
plt.plot(X_pred, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([X_pred, X_pred[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('RBF Kernel Regression on AAPL Stock Price')
plt.legend(loc='upper left')
plt.show()