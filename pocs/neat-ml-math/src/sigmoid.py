import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Call the sigmoid function with different inputs
x1 = np.linspace(-10, 10, 100)
y1 = sigmoid(x1)

x2 = np.linspace(-10, 10, 100) + 10  # Shift to the right
y2 = sigmoid(x2)

x3 = np.linspace(-10, 10, 100) + 20  # Shift further to the right
y3 = sigmoid(x3)

# Plot the results
plt.figure()
plt.plot(x1, y1, label='linspace(-10,10,100)')
plt.plot(x2, y2, label='linspace(0,20,100)')
plt.plot(x3, y3, label='linspace(10,30,100)')
plt.title('Sigmoid function for different inputs')
plt.legend()
plt.show()