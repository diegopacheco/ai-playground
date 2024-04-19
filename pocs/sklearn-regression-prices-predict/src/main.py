import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from datetime import datetime

# Load the dataset
df = pd.read_csv('stock_prices.csv')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert 'Date' to a numerical format
df['Date'] = df['Date'].map(datetime.toordinal)

# Prepare the data
X = df['Date'].values.reshape(-1,1)
y = df['Close'].values.reshape(-1,1)

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()  
model.fit(X_train, y_train)

# Make predictions using the test set
y_pred = model.predict(X_test)

# Compare the actual output values for X_test with the predicted values
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)