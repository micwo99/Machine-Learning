import numpy

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    global average, train_columns
    all_columns = ["bathrooms", "floors", "bedrooms", "yr_built", "zipcode", "price","sqft_living"]
    data = pd.DataFrame(X).drop_duplicates().dropna()
    data = data.drop(columns=["id", "date", "sqft_living15", "sqft_lot15"])
    data = data.dropna(subset=['price'])
    if y is not None:
        data = data[data["bedrooms"] < 20]
        data = data[data["sqft_lot"] < 10000000]
        for c in range(len(all_columns)):
            if c < 3:
                data = data[data[all_columns[c]] >= 0]
            else:
                data = data[data[all_columns[c]] > 0]

        data = pd.get_dummies(data, prefix="zipcode", columns=["zipcode"])
        y = data["price"]
        X = data.drop(columns=["price"])
        average = X.mean()
        train_columns = X.columns
        return X, y
    else:
        X = X.fillna(average)
        X = pd.get_dummies(X, prefix="zipcode", columns=["zipcode"])
        X = X.reindex(columns=train_columns, fill_value=0)
        return X, None


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for f in X:
        cov = np.cov(X[f], y)[0, 1]

        sigma = (np.std(X[f]) * np.std(y))
        rho = cov / sigma

        fig = px.scatter(pd.DataFrame({'x': X[f], 'y': y}), x="x", y="y", trendline="ols",
                         title=f"Correlation Between {f} Values and Response <br>Pearson Correlation {rho}",
                         labels={"x": f"{f} Values", "y": "Response Values"})
        fig.write_image(output_path + f"/pearson.correlation.{f}.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    price = df["price"]

    # Question 1 - split data into train and test sets
    train_x, train_y, test_x, test_y = split_train_test(df,price)

    # Question 2 - Preprocessing of housing prices dataset
    train_x, train_y = preprocess_data(train_x, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_x, train_y, "../ex2/")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_x = preprocess_data(test_x, None)[0]
    losses = np.zeros((91, 10))
    percent = [k for k in range(10, 101)]
    for p in percent:
        for i in range(10):
            train = train_x.sample(frac=p / 100)
            linear = LinearRegression(include_intercept=True)
            linear.fit(train, train_y.loc[train.index])
            losses[p - 10, i] = linear.loss(test_x.to_numpy(),  test_y.to_numpy())
    loss_avg = np.mean(losses, axis=1)
    std = np.std(losses, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percent, y=loss_avg, mode='lines+markers', name='Mean loss'))
    fig.add_trace(
        go.Scatter(x=percent, y=loss_avg - 2 * std, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(x=percent, y=loss_avg + 2 * std, mode='lines', line=dict(width=0), fill='tonexty',
                   fillcolor='rgba(0,100,80,0.2)', name='Std $\sigma$'))
    fig.update_layout(title='Mean loss as function of the sample size', xaxis_title='p%', yaxis_title='Loss',
                      legend_title='').show()
