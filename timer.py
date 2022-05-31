import time
import functools


def time_loger(log_list):
    def timer(func):
        @functools.wraps(func)
        def clocked(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            time_cost = time.time() - start_time
            log_list[0] += time_cost
            return result
        return clocked
    return timer
