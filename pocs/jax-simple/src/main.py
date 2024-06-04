import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Define a simple function
def f(x):
    return jnp.sin(x)

# Compute the gradient of the function
grad_f = jax.grad(f)

# Evaluate the function and its gradient at an array of x values
x = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 1000)  # Array of x values
y = f(x)

# Compute the gradient at each x value
dy_dx = jnp.array([grad_f(xi) for xi in x])

# Plot results
plt.figure(figsize=(12, 6))

# Plot function
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Function')

# Plot gradient
plt.subplot(1, 2, 2)
plt.plot(x, dy_dx)
plt.title('Gradient')

plt.show()