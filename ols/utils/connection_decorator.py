"""Decocator that makes sure the object is 'connected' according to it's connected predicate."""

from typing import Any, Callable


def connection(f: Callable) -> Callable:
    """Decocator that makes sure the object is 'connected' according to it's connected predicate.

    Example:
    ```python
    @connection
    def list_history(self) -> list[str]:
       pass
    ```
    """

    def wrapper(connectable: Any, *args: Any, **kwargs: Any) -> Callable:
        if not connectable.connected():
            connectable.connect()
        return f(connectable, *args, **kwargs)

    return wrapper
