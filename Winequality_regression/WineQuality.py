import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def ols_cost(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(x))


def gradient_descent(x, y, theta, alpha, iters):
    """

    :param x: training data
    :param y: target value
    :param theta: weights
    :param alpha: learning rate
    :param iters: number of updates
    :return:
    """

    # Defining temp matrix for theta
    temp = np.matrix(np.zeros(theta.shape))

    # Number of parameters to iter through
    parameters = int(theta.ravel().shape[1])

    # cost vector to see how it progresses through each step
    cost = np.zeros(iters + 1)
    cost[0] = ols_cost(x, y, theta)

    # Error calculation for every step
    # for i in range(iters):
    #     error


df = pd.read_csv('winequality-red.csv')

correlations = df.corr()['quality'].abs()

sns.heatmap(df.corr())          # Creates correaltion heatmap of all columns
plt.show()

corr_var = np.where(correlations > 0.3) # TODO add same condition for < -0.3

x = df[df.columns[corr_var]]
y = df['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.99, random_state=114)

reg = LinearRegression()
reg.fit(x_train, y_train)

test_predict = reg.predict(x_test)
# print("Test prediction:\n {}".format(test_predict))

# for i in range(len(y_test)):
#     print("{} : {}".format(y_test.values[i],
#                            test_predict[i]))

print(metrics.mean_squared_error(y_test, test_predict))
