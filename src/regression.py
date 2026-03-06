import math
from collections import namedtuple
from .correlation import covariance
from .core import mean


RegressionModel = namedtuple(
    "RegressionModel",
    ["slope", "intercept", "r_squared", "mse", "predictions", "residuals"]
)


def linear_regression(x, y):
    x_list = list(x)
    y_list = list(y)
    if len(x_list) != len(y_list):
        raise ValueError("x and y must have the same length")
    if not x_list:
        raise ValueError("linear_regression requires at least one data point")
    n = len(x_list)
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n
    slope = covariance(x_list, y_list, ddof=0) / (
        sum((xi - mean_x) ** 2 for xi in x_list) / n
    ) if n > 1 else 0.0
    if n == 1:
        slope = 0.0
    intercept = mean_y - slope * mean_x
    predictions = [slope * xi + intercept for xi in x_list]
    residuals = [yi - pred for yi, pred in zip(y_list, predictions)]
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yi - mean_y) ** 2 for yi in y_list)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    mse = ss_res / n if n > 0 else 0.0
    return RegressionModel(slope, intercept, r_squared, mse, predictions, residuals)


def predict(model, new_x):
    if not isinstance(model, RegressionModel):
        raise TypeError("model must be a RegressionModel")
    if isinstance(new_x, (int, float)):
        return model.slope * new_x + model.intercept
    return (model.slope * x + model.intercept for x in new_x)
