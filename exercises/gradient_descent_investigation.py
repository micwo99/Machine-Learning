import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc
from IMLearn.model_selection import cross_validate

from utils import custom


import plotly.graph_objects as go

LAMBDAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])

    return callback, values, weights


def fixed_lr(init,L,eta,figure,losses):
    callback, values, weights = get_gd_state_recorder_callback()
    gradient = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
    l = L(init)
    gradient.fit(l, None, None)

    title=str(L)[-4:-2] + " with eta = " + str(eta)
    plot_descent_path(L, np.array(weights), title=title).show()
    figure.add_traces(
        [go.Scatter(x=np.arange(1, len(values) + 1), y=values, name=title)])
    losses[eta][str(L)[-4:-2]] = l.compute_output()


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    losses = pd.DataFrame({eta: [np.inf, np.inf] for eta in etas}, index=[str(L1)[-4:-2], str(L2)[-4:-2]])
    fig = go.Figure()
    for eta in etas:
        fixed_lr(init,L1,eta,fig,losses)
        fixed_lr(init,L2,eta,fig,losses)
    fig.update_layout(title="Convergence rate for all learning rates")
    fig.show()
    print(losses)

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)





def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    pred_prob = lg.predict_proba(X_train)

    fpr, tpr, thresholds = roc_curve(y_train, pred_prob)
    c = [custom[0], custom[-1]]

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(showlegend=False,
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    index=np.argmax(tpr - fpr)
    best_alpha = thresholds[index]
    lg.alpha_ = best_alpha
    print("Best alpha: " + str(best_alpha))
    lg_test_error = lg._loss(X_test, y_test)
    print("Best lambda models test loss " + str(lg_test_error))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    q10_q11(X_test, X_train, y_test, y_train)



def q10_q11(X_test, X_train, y_test, y_train):
    alpha = 0.5
    for penalty in ["l1", "l2"]:
        train_errors, validate_errors = [], []
        for lam in LAMBDAS:
            lam_model = LogisticRegression(penalty=penalty, lam=lam, alpha=alpha,
                                           solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)))
            train_error, val_error = cross_validate(lam_model, np.array(X_train), np.array(y_train),
                                                            misclassification_error)
            train_errors.append(train_error)
            validate_errors.append(val_error)

        best_lambda = LAMBDAS[np.argmin(validate_errors)]
        best_lam_model = LogisticRegression(penalty=penalty, lam=best_lambda, alpha=alpha,
                                            solver=GradientDescent(max_iter=20000,
                                                                   learning_rate=FixedLR(1e-4))).fit(np.array(X_train),
                                                                                                     np.array(y_train))
        test_error = best_lam_model.loss(np.array(X_test), np.array(y_test))
        print(" The best λ for "+penalty+":"+str( np.round(best_lambda, 3))+" and its test error is: "+
              str(np.round(test_error, 3)))




if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()



