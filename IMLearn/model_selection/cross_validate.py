from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # indices = np.arange(X.shape[0]) % cv
    # loss_train = np.zeros(cv)
    # loss_validation = np.zeros(cv)
    # for count in range(cv):
    #     x_val ,y_val = X[indices == count],y[indices == count]
    #     train_X, train_y = X[indices != count], y[indices != count]
    #     estimator.fit(train_X, train_y)
    #     loss_train[count] = scoring(estimator.predict(train_X), train_y)
    #     loss_validation[count] = scoring(estimator.predict(x_val), y_val)
    # return loss_train.mean(), loss_validation.mean()

    train_score, validation_score = .0, .0

    # split into folds
    ids = np.arange(X.shape[0])

    folds = np.array_split(ids, cv)

    for validation_ids in folds:
        # gather train ids
        train_ids = np.setdiff1d(ids, validation_ids)

        # fit estimator
        estimator_ = deepcopy(estimator)
        estimator_.fit(X[train_ids], y[train_ids])

        # evaluate scores
        train_score += scoring(y[train_ids], estimator_.predict(X[train_ids]))
        validation_score += scoring(y[validation_ids], estimator_.predict(X[validation_ids]))

    # return average scores
    return train_score / cv, validation_score / cv


