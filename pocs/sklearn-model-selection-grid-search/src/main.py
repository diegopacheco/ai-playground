import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Create a GridSearchCV object
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)

# Fit the grid to the data
grid.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found by grid search:")
print(grid.best_params_)

# Make predictions using the best model
grid_predictions = grid.predict(X_test)

# Print a classification report
print(classification_report(y_test, grid_predictions))

# Plot a confusion matrix
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()