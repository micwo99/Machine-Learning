import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    dataFrame = pd.read_csv(filename, parse_dates=["Date"])
    data = pd.DataFrame(dataFrame).dropna().drop_duplicates()
    data = data[data["Temp"] > -30]
    data["DayOfYear"] = 0
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data = data.drop(columns=["Day", "Date"])
    data.to_csv("data_temp.csv", index=True, sep=",")
    return data

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data[data["Country"] == "Israel"]
    scatter = px.scatter(x=israel_data["DayOfYear"], y=israel_data["Temp"],
                         color=israel_data["Year"].astype(str),
                         title=" Days of years vs Temperature in Israel",
                         labels={"x": "Days", "y": "Temperature"})
    scatter.show()

    israel_data.to_csv("data_temp_israel.csv", index=True, sep=",")
    data_monthly = israel_data.groupby("Month").agg("std")
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
               'November', 'December']

    px.bar(x=months, y=data_monthly["Temp"],
           title="Standard deviation of the temperature according to the month ",
           labels={"x": "months",
                   "y": "Temperature"}).show()


    # Question 3 - Exploring differences between countries
    data_country_month = data.groupby(["Country","Month"]).agg({"Temp": ["mean", "std"]})
    data_country_month.columns = ['means', 'std']
    data_country_month = data_country_month.reset_index()
    px.line(x=data_country_month["Month"],
            y=data_country_month["means"],
            error_y=data_country_month["std"],
            color=data_country_month["Country"],
            title="Monthly average temperature in different countries",
            labels={"x": "Months",
                    "y": "mean"}).show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_data.DayOfYear,israel_data.Temp)
    losses = list()
    for k in range(1,11):
        model = PolynomialFitting(k)
        model._fit(train_X.to_numpy(), train_y.to_numpy())
        pol_loss = model._loss(test_X.to_numpy(),test_y.to_numpy())
        losses.append(round(pol_loss, 2))

    px.bar(x=[1,2,3,4,5,6,7,8,9,10], y=losses,text=losses,
           title="Losses depending on k",
           labels={"x": "k", "y": "loss"}).show()
    # Question 5 - Evaluating fitted model on different countries
    k = np.argmin(losses) + 1
    poly_fitting = PolynomialFitting(k)
    poly_fitting.fit(israel_data.DayOfYear,israel_data.Temp)
    different_country = data.drop(data[data["Country"] == "Israel"].index)

    country,loss = np.array([(country, poly_fitting.loss(different_country[different_country["Country"] == country]["DayOfYear"],
                                                  different_country[different_country["Country"] == country]["Temp"]))
                      for country in different_country["Country"].drop_duplicates()]).T

    error_bar = px.bar(x=country, y=np.array(loss).astype(float),color=list(country),text=np.array(loss).astype(float))
    error_bar.update_layout(title="loss over countries for model fitted over israel",
                            xaxis_title="Country", yaxis_title="Loss")
    error_bar.show()