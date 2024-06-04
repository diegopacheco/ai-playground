import numpy as np
import matplotlib.pyplot as plt

# Generate a sequence of numbers from 0.1 to 10 with 100 steps in between
x = np.linspace(0.1, 10, 100)

# Compute the natural logarithm
y = np.log(x)

# Create the plot
plt.plot(x, y)

# Add a title
plt.title('Natural Logarithm Function')

# Add X and Y labels
plt.xlabel('X')
plt.ylabel('ln(X)')

# Show the grid
plt.grid(True)

# Show the plot
plt.show()