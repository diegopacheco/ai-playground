import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum(axis=0)

# Call the softmax function with different inputs
x1 = np.linspace(-10, 10, 100)
y1 = softmax(x1)

x2 = np.linspace(-10, 10, 100) + 10  # Shift to the right
y2 = softmax(x2)

x3 = np.linspace(-10, 10, 100) + 20  # Shift further to the right
y3 = softmax(x3)

# Plot the results
plt.figure()
plt.plot(x1, y1, label='linspace(-10,10,100)')
plt.plot(x2, y2, label='linspace(0,20,100)')
plt.plot(x3, y3, label='linspace(10,30,100)')
plt.title('Softmax function for different inputs')
plt.legend()
plt.show()