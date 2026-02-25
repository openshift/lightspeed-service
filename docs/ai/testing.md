# Testing Guide

Read this when writing or modifying tests in this repo.

## Test Layout

```
tests/
├── unit/          # Fast, isolated, no external services
├── integration/   # Requires real services (DB, LLM, etc.)
├── e2e/           # Requires a running OLS instance
└── benchmarks/    # Performance tests
```

Default target is `make test-unit`. Always write unit tests. Integration/e2e only when unit tests cannot cover the scenario.

## Non-Obvious Setup

**Global autouse fixture — config is reset between tests.**
`tests/unit/conftest.py` calls `config.reload_empty()` before each test. This means config state does not leak between tests, but it also means any test that needs config must set it up explicitly.

**Config-dependent tests need a fixture:**
```python
@pytest.fixture(scope="function")
def _load_config():
    config.reload_from_yaml_file("tests/config/test_app_endpoints.yaml")

@pytest.mark.usefixtures("_load_config")
def test_something_that_needs_config():
    ...
```

**Module-level config setup (import-time side effects):**
Some modules read `config` at import time. Set required config fields before importing those modules:
```python
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints.streaming_ols import some_function  # noqa: E402
```

## Async Tests

Use `@pytest.mark.asyncio` for any `async def` test. Don't forget the decorator — the test will silently pass without executing if you omit it.

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected
```

For draining async generators in tests:
```python
async def drain_generator(generator) -> list:
    return [item async for item in generator]

@pytest.mark.asyncio
async def test_generator():
    result = await drain_generator(my_async_gen())
    assert len(result) == 3
```

## Credentials and Secrets in Tests

Test credentials live in `tests/config/secret/apitoken` (content: `secret_key`) and `tests/config/secret2/apitoken` (content: `secret_key_2`). Use these paths when constructing `ProviderConfig` fixtures — don't create new secret files unless necessary.

## Provider Config Fixtures

Construct `ProviderConfig` from a dict, not keyword args:

```python
@pytest.fixture
def provider_config():
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_model_url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
```

## Mocking Patterns

```python
from unittest.mock import Mock, patch, AsyncMock

# Patch at the usage site, not the definition site
@patch("ols.app.endpoints.ols.some_dependency")
def test_something(mock_dep):
    mock_dep.return_value = "mocked"
    ...

# Async mock
@patch("ols.src.query_helpers.query_helper.load_llm")
async def test_async(mock_load_llm):
    mock_load_llm.return_value = AsyncMock()
    ...
```

## Parametrize for Variants

Use `@pytest.mark.parametrize` instead of duplicating test functions:

```python
@pytest.mark.parametrize(
    "model_name,should_have_temp",
    [
        ("gpt-4o", True),
        ("o1-mini", False),
        ("gpt-5-mini", False),
    ],
)
def test_model_params(provider_config, model_name, should_have_temp):
    provider = OpenAI(model=model_name, provider_config=provider_config)
    assert ("temperature" in provider.default_params) == should_have_temp
```

## Test Naming

Pattern: `test_<function_or_behavior>_<scenario>`

```python
def test_load_returns_chat_openai_instance(): ...
def test_params_handling_filters_unknown_params(): ...
def test_credentials_key_in_directory_handling(): ...
```

## Coverage

Target is 90%+ per test type. Run `make coverage-report` to generate an HTML report. New code that drops coverage below threshold will fail CI.
