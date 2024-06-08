import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and validation datasets
# Use 75% for training, 25% for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the base models
model1 = RandomForestClassifier(random_state=1)
model2 = GradientBoostingClassifier(random_state=1)

# Train the base models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Make predictions using the base models
val_pred1 = model1.predict(X_val)
val_pred2 = model2.predict(X_val)

# Stack the predictions for validation
stacked_val_predictions = np.column_stack((val_pred1, val_pred2))

# Define the meta-model
meta_model = LogisticRegression()

# Train the meta-model using the stacked predictions
meta_model.fit(stacked_val_predictions, y_val)

# Make predictions using the base models on the entire dataset
final_pred1 = model1.predict(X)
final_pred2 = model2.predict(X)

# Stack the final predictions
stacked_final_predictions = np.column_stack((final_pred1, final_pred2))

# Make the final prediction using the meta-model
final_predictions = meta_model.predict(stacked_final_predictions)

# Print the accuracy of the blended model
print('Accuracy:', accuracy_score(y, final_predictions))

# Plot the final predictions
plt.scatter(X[:, 0], X[:, 1], c=final_predictions, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Blended Model Predictions')
plt.show()