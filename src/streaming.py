import math
from collections import deque
import re


def read_numbers_from_file(file_path):
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            numbers = re.split(r"[,\s\t]+", line)
            for num in numbers:
                if num:
                    yield float(num)


def sliding_window(iterable, window_size):
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    window = deque(maxlen=window_size)
    for item in iterable:
        window.append(item)
        if len(window) == window_size:
            yield tuple(window)


def streaming_mean(iterable):
    count = 0
    total = 0.0
    for value in iterable:
        count += 1
        total += value
        yield total / count


def streaming_variance(iterable, ddof=0):
    n = 0
    mean_val = 0.0
    m2 = 0.0
    for x in iterable:
        n += 1
        delta = x - mean_val
        mean_val += delta / n
        delta2 = x - mean_val
        m2 += delta * delta2
        if n > ddof:
            yield m2 / (n - ddof)
        else:
            yield float("nan")


def streaming_pearson(x_iter, y_iter):
    n = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    for x, y in zip(x_iter, y_iter):
        n += 1
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x ** 2
        sum_y2 += y ** 2
        if n < 2:
            yield None
        else:
            numerator = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt(
                (n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)
            )
            if denominator == 0:
                yield None
            else:
                yield numerator / denominator
