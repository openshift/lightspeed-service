"""Test decorators."""

import time
from functools import wraps


def retry(max_attempts=3, wait_between_runs=5):
    """Construct decorator that allows to retry running selected test."""

    def retry_test_decorator(test_function):
        @wraps(test_function)
        def wrapper(*args, **kwargs):
            retry_count = 1

            while retry_count < max_attempts:
                try:
                    # try to run the test
                    return test_function(*args, **kwargs)

                except AssertionError as e:
                    # retrieve error message
                    error_message, _ = e.__str__().split("\n")
                    print(
                        f'Retry error: "{test_function.__name__}": {error_message}. '
                        f"[{retry_count}/{max_attempts - 1}] "
                        f"Retrying new execution in {wait_between_runs} second(s)"
                    )
                    # time for OLS or LLM to breath
                    time.sleep(wait_between_runs)
                    retry_count += 1

            # preserve original traceback in case assertion Failed
            return test_function(*args, **kwargs)

        return wrapper

    return retry_test_decorator
