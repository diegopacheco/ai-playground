import numpy as np
import matplotlib.pyplot as plt

def dot_product(v1, v2):
    return np.dot(v1, v2)

# Define two vectors
v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([6, 7, 8, 9, 10])

# Calculate dot product
dot_prod = dot_product(v1, v2)

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(v1, label='Vector 1 (v1)', marker='o')
plt.plot(v2, label='Vector 2 (v2)', marker='o')
plt.axhline(y=dot_prod, color='r', linestyle='--', label=f'Dot Product: {dot_prod}')
plt.title('Vectors and Their Dot Product', fontsize=16)
plt.xlabel('Index', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()