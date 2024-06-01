import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Convert the dataset to a DataFrame for easier plotting
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target variable to the DataFrame
df['species'] = iris.target

# Map the target variable to species names for better visualization
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Plot the Gaussian distribution of each feature in one chart
sns.pairplot(df, hue='species', diag_kind='kde')
plt.show()