import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the California housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Perform feature engineering
# Create a new feature that represents the number of rooms per household
df['rooms_per_household'] = df['AveRooms'] / df['HouseAge']

# Create a new feature that represents the number of bedrooms per room
df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']

# Create interaction terms for the features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
df_poly = poly.fit_transform(df.drop('MedHouseVal', axis=1))

# Get the feature names after dropping 'MedHouseVal'
feature_names = df.drop('MedHouseVal', axis=1).columns

df_poly = pd.DataFrame(df_poly, columns=poly.get_feature_names_out(feature_names))

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(df_poly, df['MedHouseVal'], test_size=0.2, random_state=42)

# Define the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Print the model's RMSE score
print('RMSE Score:', np.sqrt(mean_squared_error(y_test, y_pred)))

# Plot the predicted vs actual prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()