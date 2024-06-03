import numpy as np
import matplotlib.pyplot as plt

# Generate an array of numbers in the range (-1, 1)
x = np.linspace(-0.999, 0.999, 1000)

# Compute the arctanh values
y = np.arctanh(x)

# Plot the arctanh function
plt.plot(x, y)
plt.title('Inverse Hyperbolic Tangent Function')
plt.xlabel('x')
plt.ylabel('arctanh(x)')
plt.grid(True)
plt.show()