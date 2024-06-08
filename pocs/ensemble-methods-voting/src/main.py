import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base models
model1 = RandomForestClassifier(random_state=1)
model2 = GradientBoostingClassifier(random_state=1)

# Define the voting ensemble
voting_clf = VotingClassifier(estimators=[('rf', model1), ('gb', model2)], voting='soft')

# Train the voting ensemble
voting_clf.fit(X_train, y_train)

# Make predictions using the voting ensemble
y_pred = voting_clf.predict(X_test)

# Print the accuracy of the voting ensemble
print('Accuracy:', accuracy_score(y_test, y_pred))

# Plot the predictions
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Voting Ensemble Predictions')
plt.show()