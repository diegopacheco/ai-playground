from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train AdaBoost Classifier
classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=42)
model = classifier.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('AdaBoost \nAccuracy:{0:.3f}'.format(np.mean(y_pred == y_test)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()