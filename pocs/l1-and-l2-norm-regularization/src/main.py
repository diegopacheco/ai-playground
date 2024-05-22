import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 300)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 300)

# Reshape x to be a 2D array
x = x[:, np.newaxis]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Perform Lasso regression (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(x_train, y_train)

# Perform Ridge regression (L2 regularization)
ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)

# Calculate and print the L1 norm of the Lasso coefficients
l1_norm = norm(lasso.coef_, 1)
print(f"L1 norm of Lasso coefficients: {l1_norm}")

# Calculate and print the L2 norm of the Ridge coefficients
l2_norm = norm(ridge.coef_, 2)
print(f"L2 norm of Ridge coefficients: {l2_norm}")

# Plot the L1 and L2 norms
plt.bar(['L1 norm (Lasso)', 'L2 norm (Ridge)'], [l1_norm, l2_norm])
plt.title('L1 and L2 Norms of Coefficients')
plt.ylabel('Norm')
plt.show()