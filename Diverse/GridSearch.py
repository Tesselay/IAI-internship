import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split, cross_val_score


def svc_gs(grid_size, test_data, cross_val=True, c_list=[1], gamma_list=['auto'], cv_list=[5], kernel='rbf'):
    """
    Gridsearch for Support Vector Classifier

    :param grid_size:
    :param test_data:
    :param cross_val:
    :param c_list:
    :param gamma_list:
    :param cv_list:
    :param kernel:
    :return:
    """
    x = test_data.data
    y = test_data.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=114)

    # duplicates lone value in given list that is not used for testing. Easy way to make following iterations work right.
    if len(c_list) == 1:
        c_list *= 10
    elif len(gamma_list) == 1:
        gamma_list *= 10
    elif len(cv_list) == 1:
        cv_list *= 10

    if cross_val:
        clf_list = []
        for i in range(grid_size):
            clf = svm.SVC(kernel=kernel, C=c_list[i], gamma=gamma_list[i]).fit(x_train, y_train)
            clf_list.append(clf)

        return cross_val_gridsearch(grid_size, clf_list, x, y, cv_list)
    else:
        clf_list = []
        for i in range(grid_size):
            temp_list = []
            for j in range(grid_size):
                clf = svm.SVC(kernel=kernel, C=c_list[i], gamma=gamma_list[j]).fit(x_train, y_train)
                temp_list.append(clf)
            clf_list.insert(0, temp_list)

        return acc_gridsearch(grid_size, clf_list, x_test, y_test)


def xgboost_gs(grid_size, test_data, cross_val=True, booster='gbtree', lr_list=[0.3], gamma_list=[0]):
    """
    Gridsearch for XGBoost Classifier.

    :param grid_size:
    :param test_data:
    :param cross_val:
    :param booster:
    :param lr_list:
    :param gamma_list:
    :return:
    """
    x = test_data.data
    y = test_data.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=114)

    if len(lr_list) == 1:
        for i in range(grid_size):
            lr_list.append(lr_list[0])
    elif len(gamma_list) == 1:
        for i in range(grid_size):
            gamma_list.append(gamma_list[0])

    clf_list = []
    for i in range(grid_size):
        temp_list = []
        for j in range(grid_size):
            clf = XGBClassifier(booster=booster, learning_rate=lr_list[i], gamma=gamma_list[j]).fit(x_train, y_train)
            temp_list.append(clf)
        clf_list.insert(0, temp_list)

    return acc_gridsearch(grid_size, clf_list, x_test, y_test)


def acc_gridsearch(grid_size, clf_list, x_test, y_test):
    """
    Gridsearch for accuracy values.

    :param grid_size:
    :param clf_list:
    :param x_test:
    :param y_test:
    :return:
    """
    acc_scores = []
    for i in range(grid_size):
        temp_list = []
        for j in range(grid_size):
            accuracy = clf_list[i][j].score(x_test, y_test)
            temp_list.append(np.around(accuracy, 2))
        acc_scores.insert(0, temp_list)

    return np.array(acc_scores)


def cross_val_gridsearch(grid_size, clf_list, x, y, cv_list):
    """
    Gridsearch for cross-validated values.

    :param grid_size:
    :param clf_list:
    :param x:
    :param y:
    :param cv_list:
    :return:
    """

    val_scores = []
    for i in range(grid_size):
        temp_list = []
        for j in range(grid_size):
            cross_score = cross_val_score(clf_list[i], x, y, cv=cv_list[j]).mean()
            temp_list.append(np.around(cross_score, 2))
        val_scores.insert(0, temp_list)

    return np.array(val_scores)


def heatmap(y_axis, x_axis, grid_values, title, cmap_color='Wistia'):
    """
    Function that creates heatmap of gridsearch.

    :param y_axis:
    :param x_axis:
    :param grid_values:
    :param title:
    :param cmap_color:
    :return:
    """

    fig, ax = plt.subplots()
    im = ax.imshow(grid_values)
    im.set_cmap(cmap_color)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Accuracy", rotation="-90", va="bottom")

    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_yticks(np.arange(len(y_axis)))

    ax.set_xticklabels(x_axis)
    ax.set_yticklabels(y_axis)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            ax.text(j, i, grid_values[i, j], ha="center", va="center", color="black")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()


iris = datasets.load_iris()
# Test values/value-lists:
grid_size = 10
svc_c = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 20000, 50000]
svc_gamma = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
svc_kernel = ['rbf', 'linear', 'sigmoid']
title = "Accuracy of svc-model\n(x-axis: gamma; y-axis: C; kernel: rbf)"

grid = svc_gs(grid_size=grid_size, test_data=iris, cross_val=False, c_list=svc_c, gamma_list=svc_gamma, cv_list=[10],
              kernel='rbf')
heatmap(y_axis=svc_c, x_axis=svc_gamma, grid_values=grid, title=title, cmap_color='Wistia')

iris = datasets.load_iris()
grid_size = 11
booster = ['gbtree', 'dart', 'gblinear']
xgboost_learning_rate = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 1, 10, 100]
xgboost_gamma = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
xgboost_max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
title = "Accuracy of XGBoost\n(x-axis: gamma; y-axis: learning rate; booster: gbtree)"

grid = xgboost_gs(grid_size=grid_size, test_data=iris, cross_val=False, booster=booster[0],
                  lr_list=xgboost_learning_rate, gamma_list=xgboost_gamma)
heatmap(y_axis=xgboost_learning_rate, x_axis=xgboost_gamma, grid_values=grid, title=title, cmap_color='Wistia')

# Low Lambda/Alpha/Subsample can make a small difference
