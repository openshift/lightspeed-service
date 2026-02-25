# Configuration Model Guide

Read this when adding new configuration fields, new config classes, or modifying `ols/app/models/config.py`.

## Key Facts

- All config classes use Pydantic `BaseModel`
- Config is loaded from YAML via `ProviderConfig(data_dict)` and similar constructors — not from keyword args directly
- Validation errors must raise `checks.InvalidConfigurationError`, not Pydantic's built-in errors
- `extra="forbid"` is used on provider-specific configs — unknown fields cause a hard error
- Credentials are never stored in the YAML directly; they are read from files via `checks.read_secret()`

## Config Class Patterns

### Standard field with validation

```python
class MyConfig(BaseModel):
    name: str
    optional_field: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value:
            raise checks.InvalidConfigurationError("name must not be empty")
        return value
```

### Cross-field validation

```python
@model_validator(mode="after")
def validate_consistency(self) -> Self:
    if self.field_a and not self.field_b:
        raise checks.InvalidConfigurationError("field_b is required when field_a is set")
    return self
```

### Before-mode validator (for input transformation)

```python
@model_validator(mode="before")
@classmethod
def transform_inputs(cls, data: Any) -> Any:
    data["credentials"] = checks.read_secret(
        data, constants.CREDENTIALS_PATH_SELECTOR, constants.API_TOKEN_FILENAME
    )
    return data
```

## Reading Secrets

Credentials come from files, never from the YAML values directly:

```python
# Read API token from a file path specified in config
credentials = checks.read_secret(
    data,                               # the raw config dict
    constants.CREDENTIALS_PATH_SELECTOR,  # key = "credentials_path"
    constants.API_TOKEN_FILENAME,         # filename = "apitoken"
)
```

If `credentials_path` points to a directory, it reads `<directory>/apitoken`. If it points directly to a file, it reads that file. See `ols/utils/checks.py` for the full implementation.

## Adding a New Provider-Specific Config

1. Subclass `ProviderSpecificConfig` (which provides `url`, `token`, `api_key`):

```python
class MyProviderConfig(ProviderSpecificConfig, extra="forbid"):
    credentials_path: str
    my_specific_field: Optional[str] = None
```

2. Add an optional field to `ProviderConfig`:

```python
class ProviderConfig(BaseModel):
    ...
    my_provider_config: Optional[MyProviderConfig] = None
```

3. Add a `case` branch in `ProviderConfig.set_provider_specific_configuration()`:

```python
case constants.PROVIDER_MYPROVIDER:
    my_config = data.get("my_provider_config")
    self.check_provider_config(my_config)
    self.read_api_key(my_config)
    self.my_provider_config = MyProviderConfig(**my_config)
```

4. The config key in YAML follows the pattern `<provider_type>_config` (e.g. `openai_config`, `watsonx_config`).

## Config Validation Flow

Config classes have two validation phases:

1. **`__init__`** — Pydantic field validation + any custom `__init__` logic (reading secrets, setting derived fields)
2. **`validate_yaml()`** — Called explicitly after full config load; used for cross-object validation (e.g. TLS cert file existence checks)

If you add a new config class that needs file existence checks or cross-section validation, add a `validate_yaml()` method and call it from the appropriate parent config's `validate_yaml()`.

## Important Constants

Config-related constants live in `ols/constants.py`:

| Constant | Value | Used for |
|---|---|---|
| `CREDENTIALS_PATH_SELECTOR` | `"credentials_path"` | Key for secret file path in config dicts |
| `API_TOKEN_FILENAME` | `"apitoken"` | Default filename inside credentials directory |
| `DEFAULT_CONTEXT_WINDOW_SIZE` | (int) | Default if not set in model config |
| `DEFAULT_MAX_TOKENS_FOR_RESPONSE` | (int) | Default token limit |
| `SUPPORTED_PROVIDER_TYPES` | frozenset | Validated against in `ProviderConfig.set_provider_type()` |
| `SUPPORTED_AUTHENTICATION_MODULES` | frozenset | Validated in `AuthenticationConfig` |
