"""An interface to be satisfied by all auth. implementations."""

from abc import ABC, abstractmethod

from fastapi import HTTPException, Request


def extract_bearer_token(header: str) -> str:
    """Extract the bearer token from an HTTP authorization header.

    Args:
        header: The authorization header containing the token.

    Returns:
        The extracted token if present, else an empty string.
    """
    try:
        scheme, token = header.split(" ", 1)
        return token if scheme.lower() == "bearer" else ""
    except ValueError:
        return ""


def extract_token_from_request(request: Request) -> str:
    """Extract the bearer token from a FastAPI request."""
    authorization_header = request.headers.get("Authorization")
    if not authorization_header:
        raise HTTPException(
            status_code=401, detail="Unauthorized: No auth header found"
        )
    return extract_bearer_token(authorization_header)


class AuthDependencyInterface(ABC):
    """An interface to be satisfied by all auth. implementations."""

    @abstractmethod
    async def __call__(self, request: Request) -> tuple[str, str, bool, str]:
        """Validate FastAPI Requests for authentication and authorization."""
