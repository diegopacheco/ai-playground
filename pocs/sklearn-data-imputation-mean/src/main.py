import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Print the number of missing values in each column
print("Number of missing values before imputation:")
print(titanic.isnull().sum())

# Create a SimpleImputer object
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Perform mean imputation on the 'age' column
titanic['age'] = imputer.fit_transform(titanic['age'].values.reshape(-1, 1))

# Print the number of missing values in each column after imputation
print("\nNumber of missing values after imputation:")
print(titanic.isnull().sum())

# Plot the 'age' column after imputation
plt.hist(titanic['age'], bins=20, color='c')
plt.title('Age distribution after imputation')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()