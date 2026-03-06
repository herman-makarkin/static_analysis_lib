from .exceptions import StatisticsError
from .core import median


def detect_outliers_iqr(data, k=1.5):
    data_list = list(data)
    n = len(data_list)
    if n < 4:
        raise StatisticsError("detect_outliers_iqr requires at least 4 data points")
    sorted_data = sorted(data_list)
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    q1 = sorted_data[q1_idx]
    q3 = sorted_data[q3_idx]
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    outlier_indices = [
        i for i, val in enumerate(data_list)
        if val < lower_bound or val > upper_bound
    ]
    return outlier_indices


def remove_outliers(data, method="iqr", **kwargs):
    data_list = list(data)
    if method != "iqr":
        raise ValueError(f"Unknown method: {method}")
    if len(data_list) < 4:
        for val in data_list:
            yield val
        return
    outlier_indices = set(detect_outliers_iqr(data_list, **kwargs))
    for i, val in enumerate(data_list):
        if i not in outlier_indices:
            yield val
