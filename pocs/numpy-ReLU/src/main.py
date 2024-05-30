import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Generate a range of values from -10 to 10
x = np.linspace(-10, 10, 1000)

# Apply the ReLU function to these values
y = relu(x)

# Plot the results
plt.plot(x, y)
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True)
plt.show()