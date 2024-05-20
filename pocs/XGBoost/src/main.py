from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)

# create model instance
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

bst.fit(X_train, y_train)
preds = bst.predict(X_test)
print(preds)

import matplotlib.pyplot as plt
from xgboost import plot_tree
plot_tree(bst, num_trees=0)
plt.show()

