import numpy as np
from scipy.linalg import solve

# Define the coefficients matrix (A) and the constants matrix (b)
A = np.array([[3, 2], [2, 4]])
b = np.array([7, 10])

# Solve the system of linear equations Ax = b
x = solve(A, b)

# Print the solution
print(x)

# Result
# [1. 2.]