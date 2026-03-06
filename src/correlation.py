import math
from .core import mean, std


def covariance(x, y, ddof=1):
    x_list = list(x)
    y_list = list(y)
    if len(x_list) != len(y_list):
        raise ValueError("x and y must have the same length")
    if not x_list:
        raise ValueError("covariance requires at least one data point")
    n = len(x_list)
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n
    total = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_list, y_list))
    return total / (n - ddof)


def pearson_correlation(x, y):
    x_list = list(x)
    y_list = list(y)
    if len(x_list) != len(y_list):
        raise ValueError("x and y must have the same length")
    if len(x_list) < 2:
        raise ValueError("pearson_correlation requires at least two data points")
    n = len(x_list)
    sum_x = sum(x_list)
    sum_y = sum(y_list)
    sum_xy = sum(xi * yi for xi, yi in zip(x_list, y_list))
    sum_x2 = sum(xi ** 2 for xi in x_list)
    sum_y2 = sum(yi ** 2 for yi in y_list)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
    if denominator == 0:
        raise ValueError("standard deviation is zero")
    return numerator / denominator
