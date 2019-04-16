import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz as gv
from sklearn import datasets, neighbors, svm, tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

iris = datasets.load_iris()

df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# def dtc(criterion, ):
#     x = test_data.data
#     y = test_data.target
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=114)
#
#     clf = tree.DecisionTreeClassifier()
#     clf = clf.fit(x_train, y_train)
#     accuracy = clf.score(x_test, y_test)
#     scores = cross_val_score(clf, x, y, cv=5)
#
#     return scores.mean()


# def knn(neighbors, p, test_data, weight='uniform'):
#     x = test_data.data
#     y = test_data.target
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=114)
#
#     clf = neighbors.KNeighborsClassifier(n_neighbors=neighbors, p=p, weights=weight)
#     clf.fit(x_train, y_train)
#     accuracy = clf.score(x_test, y_test)
#     scores = cross_val_score(clf, x, y, cv=5)
#
#     return scores.mean()


# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 feature_names=iris.feature_names,
#                                 class_names=iris.target_names,
#                                 filled=True,
#                                 rounded=True,
#                                 special_characters=True)
# graph = gv.Source(dot_data)
# graph.render("iris")

_, ax = plt.subplots()
groups = df.groupby('target')

for name, group in groups:
    ax.plot(group['sepal length (cm)'], group['sepal width (cm)'], marker='o', linestyle='', ms=4, label=iris.target_names[int(name)])
ax.legend()
plt.title("Iris Flower - Sepal Size to species classification")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.show()

_, ax = plt.subplots()
groups = df.groupby('target')

for name, group in groups:
    ax.plot(group['petal length (cm)'], group['petal width (cm)'], marker='o', linestyle='', ms=4, label=iris.target_names[int(name)])
ax.legend()
plt.title("Iris Flower - Petal Size to species classification")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.show()
