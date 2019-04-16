from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=97)

clf = XGBClassifier(learning_rate=1, booster='gbtree', gamma=0.1)
clf.fit(x_train, y_train)

accuracy = clf.score(x_train, y_train)
print("Accuracy: {}".format(accuracy))

