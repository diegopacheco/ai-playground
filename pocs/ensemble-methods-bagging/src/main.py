import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the bagging classifier without specifying base_estimator
bagging_clf = BaggingClassifier(n_estimators=10, random_state=42)

# Train the bagging classifier
bagging_clf.fit(X_train, y_train)

# Make predictions using the bagging classifier
y_pred = bagging_clf.predict(X_test)

# Print the accuracy of the bagging classifier
print('Accuracy:', accuracy_score(y_test, y_pred))

# Plot the predictions
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Bagging Classifier Predictions')
plt.show()