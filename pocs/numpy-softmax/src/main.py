import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum()

# Generate a range of values from -10 to 10
x = np.linspace(-10, 10, 1000)

# Apply the softmax function to these values
y = softmax(x)

# Plot the results
plt.plot(x, y)
plt.title('Softmax Function')
plt.xlabel('x')
plt.ylabel('Softmax(x)')
plt.grid(True)
plt.show()