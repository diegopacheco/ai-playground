from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

# Loading the Iris model
X, y = load_iris(return_X_y=True)
y[y != 1] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Comparing SVC and most_frequent
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
print(f"SVC score: {clf.score(X_test, y_test)}")

clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(f"DummyClassifier score: {clf.score(X_test, y_test)}")