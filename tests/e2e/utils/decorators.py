"""Test decorators."""

import time
from functools import wraps


def retry(
    max_attempts=3,
    wait_between_runs=5,
    on_error=AssertionError,
    expected_error_message="",
):
    """Construct decorator that allows to retry running selected test.

    Args:
        max_attempts: how many times the test will be rerun in case of failure
        wait_between_runs: time duration between consecutive test runs
        on_error: expected error type (other errors will mark test failure immediatelly)
        expected_error_message: text that must be contained in error message
                                (if not, the test failure will be mark immediatelly)

    Returns:
        decorated test function
    """

    def retry_test_decorator(test_function):
        @wraps(test_function)
        def wrapper(*args, **kwargs):
            retry_count = 1

            while retry_count < max_attempts:
                try:
                    # try to run the test
                    return test_function(*args, **kwargs)

                except Exception as e:
                    if not isinstance(e, on_error):
                        raise

                    # retrieve error message
                    error_message = str(e).strip().split("\n")[0]

                    if expected_error_message not in error_message:
                        raise

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
