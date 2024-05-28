import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Load the data
data = pd.read_csv('apple_stock_prices.csv')

# Set up the figure and axis
fig, ax = plt.subplots()

# Initialize the plot
ax.plot(data['Date'], data['Close'])

# Define the animation function
def animate(i):
	ax.clear()
	ax.plot(data['Date'][:i], data['Close'][:i])
	ax.set_xlabel('Date')
	ax.set_ylabel('Closing Price ($)')
	ax.set_title('Apple Stock Price Movement')

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(data), interval=50)

# Save the animation to a video file
ani.save('apple_stock_price_movement.mp4', writer='ffmpeg')

# Show the animation
plt.show()