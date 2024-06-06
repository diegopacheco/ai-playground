import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Download historical data as dataframe
data = yf.download('AAPL', start='2020-01-01', end='2022-12-31')

# Use only closing prices
prices = data['Close']

# Calculate the daily returns
returns = prices.pct_change().dropna()

# Create histogram of the data
plt.hist(returns, bins=30, density=True, alpha=0.6, color='g')

# Fit a normal distribution to the data
mu, std = norm.fit(returns)

# Plot the PDF
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.6f,  std = %.6f" % (mu, std)
plt.title(title)

plt.show()