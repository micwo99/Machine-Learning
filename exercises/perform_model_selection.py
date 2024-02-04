from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    df = datasets.load_diabetes()
    X, y = df['data'], df['target']
    shuffle_indices = np.random.permutation(len(X))
    train_ind = shuffle_indices[:n_samples]
    test_ind = shuffle_indices[n_samples:]
    train_X, train_y = X[train_ind], y[train_ind]
    test_X, test_y = X[test_ind], y[test_ind]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    avg_train_ridge , avg_valid_ridge, avg_train_lasso, avg_valid_lasso= np.zeros(n_evaluations),np.zeros(n_evaluations)\
        , np.zeros(n_evaluations), np.zeros(n_evaluations)
    lambdas = np.linspace(0, 10, num=n_evaluations + 1)[1:]
    for i in range(len(lambdas)):
        avg_train_ridge[i], avg_valid_ridge[i] = cross_validate(RidgeRegression(lambdas[i], True), train_X, train_y,
                                                                             mean_square_error)
        avg_train_lasso[i], avg_valid_lasso[i] = cross_validate(Lasso(lambdas[i]), train_X, train_y,
                                                                             mean_square_error)
    go.Figure(
        [go.Scatter(x=lambdas, y=avg_train_ridge, mode="markers+lines", name="Average train error Ridge",
                    marker=dict(color="blue")),
         go.Scatter(x=lambdas, y=avg_valid_ridge, mode="markers+lines",
                    name="Average validation error Ridge",
                    marker=dict(color="green")),
         go.Scatter(x=lambdas, y=avg_train_lasso, mode="markers+lines",
                    name="Average train error Lasso",
                    marker=dict(color="red")),
         go.Scatter(x=lambdas, y=avg_valid_lasso, mode="markers+lines",
                    name="Average validation error Lasso",
                    marker=dict(color="orange"))
         ]).update_layout(
        title="5-fold Cross-validation using different methods", xaxis=dict(title="lambda value"),
        yaxis=dict(title="Error")).show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge , lass= lambdas[int(np.argmin(avg_valid_ridge))], lambdas[int(np.argmin(avg_valid_lasso))]
    reg = RidgeRegression(ridge, True).fit(train_X, train_y)
    lasso = Lasso(lass).fit(train_X, train_y)
    least_square = LinearRegression().fit(train_X, train_y)
    print("best lambda ridge= ",ridge)
    print("best lambda lasso = ",lass)
    print("Ridge = ", reg.loss(test_X, test_y))
    print("Lasso = ", mean_square_error(lasso.predict(test_X), test_y))
    print("Least Squares = ", least_square.loss(test_X, test_y))



if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()

