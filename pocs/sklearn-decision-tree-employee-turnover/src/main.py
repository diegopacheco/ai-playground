import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('employee_data.csv')

# Preprocess the data
df['satisfaction'] = df['satisfaction'].astype(float)
df['tenure'] = df['tenure'].astype(float)
df['performance'] = df['performance'].astype(float)
df['left_company'] = df['left_company'].astype(int)

# Split the data into features and target
X = df[['satisfaction', 'tenure', 'performance']]
y = df['left_company']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

def predict(satisfaction, tenure, performance):
    return clf.predict([[satisfaction, tenure, performance]])

# Test the predict function
print(predict(0.5, 5, 0.7))  # Replace with actual values

# Plot the data
plt.figure(figsize=(10, 7))
plt.scatter(df['satisfaction'], df['performance'], c=df['left_company'])
plt.xlabel('Satisfaction')
plt.ylabel('Performance')
plt.title('Employee Data')
plt.colorbar(label='Left Company')
plt.show()