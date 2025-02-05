"""An interface to be satisfied by all auth. implementations."""

from abc import ABC, abstractmethod

from fastapi import Request


class AuthDependencyInterface(ABC):
    """An interface to be satisfied by all auth. implementations."""

    @abstractmethod
    async def __call__(self, request: Request) -> tuple[str, str, bool]:
        """Validate FastAPI Requests for authentication and authorization."""
