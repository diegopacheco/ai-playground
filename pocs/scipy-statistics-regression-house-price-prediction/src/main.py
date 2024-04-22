import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Load data (replace with real data)
size = np.array([1000, 1200, 1500, 1800, 2000])
price = np.array([200000, 240000, 300000, 360000, 400000])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(size, price)

# Predict price for a new house
new_size = 2200
predicted_price = slope * new_size + intercept

# Print results
print(f"Predicted price for a {new_size} sqft house: ${predicted_price:.2f}")

# Plot data and regression line
plt.scatter(size, price)
plt.plot([0, max(size)], [intercept, slope * max(size) + intercept], 'r')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('House Price Prediction')
plt.show()
