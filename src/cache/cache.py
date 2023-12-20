from abc import ABC, abstractmethod
from typing import Union


class Cache(ABC):
    @abstractmethod
    def get(self, key: str) -> Union[str, None]:
        """Abstract method to retrieve a value from the cache.

        Args:
            key (str): The key associated with the value.

        Returns:
            Union[str, None]: The value associated with the key, or None if not found.
        """
        pass

    @abstractmethod
    def insert_or_append(self, key: str, value: str) -> None:
        """Abstract method to store a value in the cache.

        Args:
            key (str): The key to associate with the value.
            value (str): The value to be stored in the cache.

        Returns:
            None
        """
        pass
