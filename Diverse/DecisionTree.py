import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz as gv
from sklearn import datasets, preprocessing, tree
from sklearn.model_selection import train_test_split, cross_val_score

"""Functions for try at matching my own dt model to fit sklearn parameters for plotting. Didn't work out"""
def gini_index(groups, classes):
    samples = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0

        for class_val in classes:
            p = [row for row in group].count(class_val) / size
            score += p * p

        gini += (1.0 - score) * (size / samples)

    return gini


def fit(x, y, classifier=True):
    n_samples, n_features = x.shape
    y = np.atleast_1d(y)

    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))

    n_outputs = y.shape[1]

    if classifier:
        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.int)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded


"""Own decision tree model. Is overfitted to, and only will work with, the iris flower dataset"""
print("\n ~- Own Decision Tree Model -~\n")

iris = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

train, test = np.split(df.sample(frac=1), [int(0.6*len(df))])
_, validation = np.split(df.sample(frac=1), [int(0.6*len(df))])

print("Train-set size: {}".format(train['target'].size))
print("Test-set size: {}\n".format(test['target'].size))

# Calculates mean of petal length for virginica
train_virginica = train.loc[df.target == 2]
train_virginica_pl = train_virginica.loc[:, 'petal length (cm)']
virginica_pl_mean = train_virginica_pl.mean()
print("Virginica mean petal length: {}".format(virginica_pl_mean))

# Calculates mean of petal length for versicolor
train_versicolor = train.loc[df.target == 1]
train_versicolor_pl = train_versicolor.loc[:, 'petal length (cm)']
versicolor_pl_mean = train_versicolor_pl.mean()
print("Versicolor mean petal length: {}".format(versicolor_pl_mean))

# Calculates mean of petal length for setosa
train_setosa = train.loc[df.target == 0]
train_setosa_pl = train_setosa.loc[:, 'petal length (cm)']
setosa_pl_mean = train_setosa_pl.mean()
print("Setosa mean petal length: {}\n".format(setosa_pl_mean))

# Calculates absolute midpoint between neighbouring species
vir_ver_mid = virginica_pl_mean - (virginica_pl_mean - versicolor_pl_mean) / 2
ver_set_mid = versicolor_pl_mean - (versicolor_pl_mean - setosa_pl_mean) / 2
print("Midsection Virginica/Versicolor: {}".format(vir_ver_mid))
print("Midsection Versicolor/Setosa: {}".format(ver_set_mid))

# Redefines species based on midsection calculated through train data
test['target'] = np.nan
test.loc[test['petal length (cm)'] < ver_set_mid, 'target'] = 0.0
test.loc[np.logical_and(ver_set_mid < test['petal length (cm)'], test['petal length (cm)'] < vir_ver_mid), 'target'] = 1.0
test.loc[test['petal length (cm)'] > vir_ver_mid, 'target'] = 2.0

# Compares redefined test-set target values with validation-set target values and resets index
result = np.equal(test['target'], validation['target'])
result = result.reset_index(drop=True)

# Counts mismatches in comparison
errors = 0
for i in range(0, result.size):
    if not result[i]:
        errors += 1

# Calculates accuracy
accuracy = 1 - errors/df['target'].size
print("Errors: {}".format(errors))
print("Accuracy: {}".format(accuracy))


# TODO fix decision tree classifier example
# x = iris.data
# y = iris.target
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=72)
# plot_step = 0.05
# for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
#                                [1, 2], [1, 3], [2, 3]]):
#     X = x_train[:, pair]
#     Y = y_train
#
#     clf = tree.DecisionTreeClassifier().fit(X, Y)
#
#     plt.subplot(2, 3, pairidx + 1)
#  
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# 
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                          np.arange(y_min, y_max, plot_step))
#     plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
#
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     cs = plt.contourf(xx, yy, Z, cmap='RdYlBu')
#
#     plt.xlabel(iris.feature_names[pair[0]])
#     plt.ylabel(iris.feature_names[pair[1]])

    for i, color in zip(range(3), "ryb"):

        idx = np.where(y == i)

        print("idx: {}".format(idx))

        plt.scatter(X[idx, 0],
                    X[idx, 1],
                    c=color,
                    label=iris.target_names[i],
                    cmap='RdYlBu',
                    edgecolor='black',
                    s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()

