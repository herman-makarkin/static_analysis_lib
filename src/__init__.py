from .exceptions import StatisticsError
from .core import mean, median, mode, variance, std
from .correlation import covariance, pearson_correlation
from .regression import linear_regression, predict, RegressionModel
from .outliers import detect_outliers_iqr, remove_outliers
from .streaming import (
    read_numbers_from_file,
    sliding_window,
    streaming_mean,
    streaming_variance,
    streaming_pearson,
)
from .decorators import timer, log_decorator, validate_numeric, memoize
from .utils import ensure_list, is_numeric

__all__ = [
    "StatisticsError",
    "mean",
    "median",
    "mode",
    "variance",
    "std",
    "covariance",
    "pearson_correlation",
    "linear_regression",
    "predict",
    "RegressionModel",
    "detect_outliers_iqr",
    "remove_outliers",
    "read_numbers_from_file",
    "sliding_window",
    "streaming_mean",
    "streaming_variance",
    "streaming_pearson",
    "timer",
    "log_decorator",
    "validate_numeric",
    "memoize",
    "ensure_list",
    "is_numeric",
]
