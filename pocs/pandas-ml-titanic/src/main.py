import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(seed=1234)

df = pd.read_csv('data/train.csv', header=0)
print(df.head(3))

import matplotlib.pyplot as plt
# Describe features
print(df.describe())

# Correlation matrix
plt.matshow(df.corr())
continuous_features = df.describe(exclude='Name').columns
plt.xticks(range(len(continuous_features)), continuous_features, rotation="45")
plt.yticks(range(len(continuous_features)), continuous_features, rotation="45")
plt.colorbar()
plt.show()