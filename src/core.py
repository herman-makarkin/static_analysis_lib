import math
from collections import Counter
from .exceptions import StatisticsError


def mean(data):
    total = 0
    count = 0
    for value in data:
        total += value
        count += 1
    if count == 0:
        raise StatisticsError("mean requires at least one data point")
    return total / count


def median(data):
    sorted_data = sorted(list(data))
    n = len(sorted_data)
    if n == 0:
        raise StatisticsError("median requires at least one data point")
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return float(sorted_data[mid])


def mode(data):
    data_list = list(data)
    if not data_list:
        raise StatisticsError("mode requires at least one data point")
    counter = Counter(data_list)
    return counter.most_common(1)[0][0]


def variance(data, ddof=0):
    n = 0
    mean_val = 0.0
    m2 = 0.0
    for x in data:
        n += 1
        delta = x - mean_val
        mean_val += delta / n
        delta2 = x - mean_val
        m2 += delta * delta2
    if n == 0:
        raise StatisticsError("variance requires at least one data point")
    if ddof >= n:
        raise StatisticsError("ddof must be less than number of data points")
    return m2 / (n - ddof)


def std(data, ddof=0):
    return math.sqrt(variance(data, ddof))
