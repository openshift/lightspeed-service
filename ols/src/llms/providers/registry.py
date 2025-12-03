"""LLM providers registry."""

import logging
from collections.abc import Callable
from typing import ClassVar

from ols.src.llms.providers.provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMProvidersRegistry:
    """Registry for LLM providers."""

    llm_providers: ClassVar[dict[str, type[LLMProvider]]] = {}

    @classmethod
    def register(cls, provider_type: str, llm_provider: type[LLMProvider]) -> None:
        """Register LLM provider."""
        if not issubclass(llm_provider, LLMProvider):
            raise TypeError(
                f"LLMProvider subclass required, got '{type(llm_provider)}'"
            )
        cls.llm_providers[provider_type] = llm_provider
        logger.debug("LLM provider '%s' registered", provider_type)


def register_llm_provider_as(
    provider_type: str,
) -> Callable[[type[LLMProvider]], type[LLMProvider]]:
    """Register LLM provider in the `LLMProvidersRegistry`.

    Example:
    ```python
    @register_llm_provider_as("openai")
    class OpenAI(LLMProvider):
       pass
    ```
    """

    def decorator(cls: type[LLMProvider]) -> type[LLMProvider]:
        LLMProvidersRegistry.register(provider_type, cls)
        return cls

    return decorator
