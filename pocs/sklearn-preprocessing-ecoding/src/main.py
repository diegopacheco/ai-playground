import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Select a few columns to encode
data = titanic[['class', 'sex', 'embark_town']]

# Create a OneHotEncoder object
encoder = OneHotEncoder()

# Fit the encoder to the data and transform the data
data_encoded = encoder.fit_transform(data).toarray()

# Print the original data
print("Original Data:")
print(data.head())

# Print the encoded data
print("\nEncoded Data:")
print(pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(data.columns)).head())