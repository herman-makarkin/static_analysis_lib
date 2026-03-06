import pytest
import math
import tempfile
import os

from src import (
    StatisticsError,
    mean, median, mode, variance, std,
    covariance, pearson_correlation,
    linear_regression, predict, RegressionModel,
    detect_outliers_iqr, remove_outliers,
    read_numbers_from_file, sliding_window,
    streaming_mean, streaming_variance, streaming_pearson,
    timer, log_decorator, validate_numeric, memoize,
    ensure_list, is_numeric,
)


class TestCore:
    def test_mean_basic(self):
        assert mean([1, 2, 3, 4, 5]) == 3.0

    def test_mean_generator(self):
        assert mean(x for x in [1, 2, 3, 4, 5]) == 3.0

    def test_mean_empty(self):
        with pytest.raises(StatisticsError):
            mean([])

    def test_median_odd(self):
        assert median([1, 2, 3, 4, 5]) == 3.0

    def test_median_even(self):
        assert median([1, 2, 3, 4]) == 2.5

    def test_median_empty(self):
        with pytest.raises(StatisticsError):
            median([])

    def test_mode_basic(self):
        assert mode([1, 2, 2, 3, 3, 3, 4]) == 3

    def test_mode_first(self):
        assert mode([1, 1, 2, 2]) == 1

    def test_mode_empty(self):
        with pytest.raises(StatisticsError):
            mode([])

    def test_variance_population(self):
        assert abs(variance([1, 2, 3, 4, 5], ddof=0) - 2.0) < 0.0001

    def test_variance_sample(self):
        assert abs(variance([1, 2, 3, 4, 5], ddof=1) - 2.5) < 0.0001

    def test_variance_empty(self):
        with pytest.raises(StatisticsError):
            variance([])

    def test_variance_ddof_too_large(self):
        with pytest.raises(StatisticsError):
            variance([1], ddof=1)

    def test_std_basic(self):
        data = [1, 2, 3, 4, 5]
        assert abs(std(data, ddof=0) - math.sqrt(2.0)) < 0.0001


class TestCorrelation:
    def test_covariance_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert covariance(x, y) > 0

    def test_covariance_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        assert covariance(x, y) < 0

    def test_covariance_length_mismatch(self):
        with pytest.raises(ValueError):
            covariance([1, 2], [1, 2, 3])

    def test_covariance_empty(self):
        with pytest.raises(ValueError):
            covariance([], [])

    def test_pearson_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert abs(pearson_correlation(x, y) - 1.0) < 0.0001

    def test_pearson_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        assert abs(pearson_correlation(x, y) - (-1.0)) < 0.0001

    def test_pearson_length_mismatch(self):
        with pytest.raises(ValueError):
            pearson_correlation([1, 2], [1, 2, 3])

    def test_pearson_zero_std(self):
        with pytest.raises(ValueError):
            pearson_correlation([1, 1, 1], [1, 2, 3])


class TestRegression:
    def test_linear_regression_basic(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        model = linear_regression(x, y)
        assert abs(model.slope - 2.0) < 0.0001
        assert abs(model.intercept - 0.0) < 0.0001
        assert abs(model.r_squared - 1.0) < 0.0001

    def test_linear_regression_length_mismatch(self):
        with pytest.raises(ValueError):
            linear_regression([1, 2], [1, 2, 3])

    def test_linear_regression_empty(self):
        with pytest.raises(ValueError):
            linear_regression([], [])

    def test_predict_single(self):
        model = RegressionModel(slope=2.0, intercept=1.0, r_squared=1.0, mse=0.0, predictions=[], residuals=[])
        assert predict(model, 5) == 11.0

    def test_predict_multiple(self):
        model = RegressionModel(slope=2.0, intercept=1.0, r_squared=1.0, mse=0.0, predictions=[], residuals=[])
        result = list(predict(model, [1, 2, 3]))
        assert result == [3.0, 5.0, 7.0]

    def test_predict_invalid_model(self):
        with pytest.raises(TypeError):
            predict({}, 5)


class TestOutliers:
    def test_detect_outliers_basic(self):
        data = [1, 2, 3, 4, 5, 100]
        outliers = detect_outliers_iqr(data)
        assert 5 in outliers

    def test_detect_outliers_no_outliers(self):
        data = [1, 2, 3, 4, 5, 6]
        outliers = detect_outliers_iqr(data)
        assert outliers == []

    def test_detect_outliers_insufficient_data(self):
        with pytest.raises(StatisticsError):
            detect_outliers_iqr([1, 2, 3])

    def test_remove_outliers_basic(self):
        data = [1, 2, 3, 4, 5, 100]
        result = list(remove_outliers(data))
        assert 100 not in result


class TestStreaming:
    def test_read_numbers_from_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("1 2 3\n4,5,6\n7\t8\t9\n")
            fname = f.name
        try:
            numbers = list(read_numbers_from_file(fname))
            assert numbers == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        finally:
            os.unlink(fname)

    def test_sliding_window(self):
        data = [1, 2, 3, 4, 5]
        windows = list(sliding_window(data, 3))
        assert windows == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    def test_sliding_window_too_small(self):
        data = [1, 2]
        windows = list(sliding_window(data, 5))
        assert windows == []

    def test_streaming_mean(self):
        data = [1, 2, 3, 4, 5]
        means = list(streaming_mean(data))
        assert means == [1.0, 1.5, 2.0, 2.5, 3.0]

    def test_streaming_variance(self):
        data = [1, 2, 3, 4, 5]
        variances = list(streaming_variance(data, ddof=0))
        assert len(variances) == 5

    def test_streaming_pearson(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        correlations = list(streaming_pearson(x, y))
        assert correlations[0] is None
        assert abs(correlations[-1] - 1.0) < 0.0001


class TestDecorators:
    def test_timer(self, capsys):
        @timer
        def slow_func():
            return 42
        result = slow_func()
        assert result == 42
        captured = capsys.readouterr()
        assert "executed in" in captured.out

    def test_validate_numeric_pass(self):
        @validate_numeric
        def sum_values(values):
            return sum(values)
        assert sum_values([1, 2, 3]) == 6

    def test_validate_numeric_fail(self):
        @validate_numeric
        def sum_values(values):
            return sum(values)
        with pytest.raises(TypeError):
            sum_values([1, "a", 3])

    def test_validate_numeric_bool_fails(self):
        @validate_numeric
        def sum_values(values):
            return sum(values)
        with pytest.raises(TypeError):
            sum_values([1, True, 3])

    def test_memoize_basic(self):
        call_count = 0
        @memoize()
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        assert expensive_func(5) == 10
        assert expensive_func(5) == 10
        assert call_count == 1

    def test_memoize_maxsize(self):
        call_count = 0
        @memoize(maxsize=2)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        expensive_func(1)
        expensive_func(2)
        expensive_func(3)
        expensive_func(1)
        assert call_count == 4


class TestUtils:
    def test_ensure_list_list(self):
        result = ensure_list([1, 2, 3])
        assert result == [1, 2, 3]

    def test_ensure_list_generator(self):
        gen = (x for x in [1, 2, 3])
        with pytest.warns(UserWarning):
            result = ensure_list(gen)
        assert result == [1, 2, 3]

    def test_is_numeric_int(self):
        assert is_numeric(5) is True

    def test_is_numeric_float(self):
        assert is_numeric(3.14) is True

    def test_is_numeric_bool(self):
        assert is_numeric(True) is False

    def test_is_numeric_string(self):
        assert is_numeric("5") is False

    def test_is_numeric_complex(self):
        assert is_numeric(1 + 2j) is False
