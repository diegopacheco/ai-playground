import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create the RFE model and select 3 attributes
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=3)
rfe = rfe.fit(X, y)

# summarize the selection of the attributes
print('Selected features: %s' % list(iris.feature_names[i] for i in range(len(rfe.support_)) if rfe.support_[i]))

# Perform cross-validation and compute scores
scores = cross_val_score(rfe, X, y, cv=5)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(scores) + 1), scores)
plt.show()