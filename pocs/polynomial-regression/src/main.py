import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 300)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 300)

# Reshape x to be a 2D array
x = x[:, np.newaxis]

# Transform x to have polynomial features
polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

# Perform polynomial regression on the polynomial features
model = LinearRegression()
model.fit(x_poly, y)

# Generate a smooth curve for plotting the model
x_plot = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_plot_poly = polynomial_features.transform(x_plot)
y_plot = model.predict(x_plot_poly)

# Plot the original data and the polynomial fit
plt.scatter(x, y, color='blue')
plt.plot(x_plot, y_plot, color='red')
plt.title('Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()