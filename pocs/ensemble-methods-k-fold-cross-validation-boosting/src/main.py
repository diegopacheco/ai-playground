import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Initialize Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

# Perform k-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
results = cross_val_score(clf, X, y, cv=kfold)

# Print cross-validation score
print('Cross-validation score:', results.mean())

# Fit model
clf.fit(X, y)

# Plot feature importance
plt.bar(iris.feature_names, clf.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()