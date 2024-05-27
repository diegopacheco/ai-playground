import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy.stats import uniform

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter distribution
param_dist = {'C': uniform(loc=0, scale=4), 'gamma': uniform(loc=0, scale=1), 'kernel': ['rbf']}

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(SVC(), param_dist, refit=True, verbose=2, n_iter=100)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found by random search:")
print(random_search.best_params_)

# Make predictions using the best model
random_search_predictions = random_search.predict(X_test)

# Print a classification report
print(classification_report(y_test, random_search_predictions))

# Plot a confusion matrix
cm = confusion_matrix(y_test, random_search_predictions)
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()