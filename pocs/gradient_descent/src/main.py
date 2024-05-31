import numpy as np
import matplotlib.pyplot as plt

# The function we want to minimize
def f(x):
    return (x + 5) ** 2

# The derivative of the function
def df(x):
    return 2 * (x + 5)

# Gradient descent
def gradient_descent(x_start, learning_rate, n_iters):
    x = x_start
    history = np.zeros(n_iters)
    for i in range(n_iters):
        x -= learning_rate * df(x)
        history[i] = x
    return history

# Parameters
x_start = 3
learning_rate = 0.1
n_iters = 50

# Run gradient descent
history = gradient_descent(x_start, learning_rate, n_iters)

# Plot the function and the path taken by gradient descent
x = np.linspace(-10, 5, 500)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='y = (x+5)²')
plt.plot(history, f(history), 'ro-', label='Gradient Descent')
plt.title('Gradient Descent on y = (x+5)²')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()