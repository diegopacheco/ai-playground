import yfinance as yf
import pandas as pd

# Download historical stock price data
df = yf.download('AAPL', start='2020-01-01', end='2020-12-31')

# Keep only the 'Close' column
df = df[['Close']]

# Save the data to a CSV file
df.to_csv('apple_stock_prices.csv')