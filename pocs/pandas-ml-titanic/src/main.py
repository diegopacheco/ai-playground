import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(seed=1234)

df = pd.read_csv('data/train.csv', header=0)
print(df.head(3))

import matplotlib.pyplot as plt
# Describe features
print(df.describe())

# Filtering some features out first
cols = set(df.columns) - {'Name','Cabin','Embarked','Sex','Ticket'}
df_filtered = df[list(cols)]

# Correlation matrix
plt.matshow(df_filtered.corr())

continuous_features = df_filtered.describe().columns
plt.xticks(range(len(continuous_features)), continuous_features, rotation="45")
plt.yticks(range(len(continuous_features)), continuous_features, rotation="45")
plt.colorbar()
plt.show()

# Histograms
print(df["Age"].hist())

# Sorting
print(df.sort_values("Age", ascending=False).head())