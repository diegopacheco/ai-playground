import numpy as np
import matplotlib.pyplot as plt

# Generate a sequence of numbers from -10 to 10 with 1000 steps in between
x = np.linspace(-10, 10, 1000)

# Compute the sinc function
y = np.sinc(x)

# Create the plot
plt.plot(x, y)

# Add a title
plt.title('Sinc Function')

# Add X and Y labels
plt.xlabel('X')
plt.ylabel('sinc(X)')

# Show the grid
plt.grid(True)

# Show the plot
plt.show()