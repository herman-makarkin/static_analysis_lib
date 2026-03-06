import warnings
import inspect


def ensure_list(iterable):
    if inspect.isgenerator(iterable):
        warnings.warn(
            "Converting generator to list may load all data into memory",
            UserWarning
        )
    return list(iterable)


def is_numeric(value):
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and not isinstance(value, complex)
