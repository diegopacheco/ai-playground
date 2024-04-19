import pandas as pd
from datetime import datetime, timedelta

# Create a date range for the past 10 days
date_range = pd.date_range(end=datetime.now(), periods=10).to_pydatetime().tolist()

# Generate some random closing prices
close_prices = pd.np.random.randint(100, 200, size=10)

# Create a DataFrame
df = pd.DataFrame({
    'Date': date_range,
    'Close': close_prices
})

# Write the DataFrame to a CSV file
df.to_csv('stock_prices.csv', index=False)