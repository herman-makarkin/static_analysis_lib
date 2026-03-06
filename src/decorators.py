import time
import functools
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{func.__name__} executed in {elapsed:.3f} ms")
        return result
    return wrapper


def log_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_str = ", ".join(repr(a) for a in args)
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        logger.info(f"Calling {func.__name__}({all_args})")
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} returned {repr(result)}")
        return result
    return wrapper


def validate_numeric(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for item in arg:
                    if not isinstance(item, (int, float)) or isinstance(item, bool):
                        raise TypeError(
                            f"All elements must be numeric, got {type(item)}"
                        )
        for value in kwargs.values():
            if isinstance(value, (list, tuple)):
                for item in value:
                    if not isinstance(item, (int, float)) or isinstance(item, bool):
                        raise TypeError(
                            f"All elements must be numeric, got {type(item)}"
                        )
        return func(*args, **kwargs)
    return wrapper


def memoize(maxsize=None):
    def decorator(func):
        cache = OrderedDict() if maxsize else {}
        @functools.wraps(func)
        def wrapper(*args):
            if args in cache:
                if maxsize:
                    cache.move_to_end(args)
                return cache[args]
            result = func(*args)
            cache[args] = result
            if maxsize and len(cache) > maxsize:
                cache.popitem(last=False)
            return result
        wrapper.cache = cache
        return wrapper
    return decorator
