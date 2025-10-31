# OpenShift LightSpeed Service - Development Guide for Claude

## Project Overview
OpenShift LightSpeed (OLS) is an AI-powered assistant service for OpenShift built with FastAPI, LangChain, and LlamaIndex. It provides AI responses to OpenShift/Kubernetes questions using various LLM backends.

## Key Architecture
- **FastAPI/Uvicorn** - REST API server
- **LangChain/LlamaIndex** - LLM integration and RAG
- **Modular Provider System** - Support for OpenAI, Azure OpenAI, WatsonX, RHEL AI, etc.
- **Async/Await** - Throughout the codebase
- **Pydantic Models** - Configuration and data validation

## Code Standards

### Python Version & Dependencies
- **Python 3.11/3.12** - Target version py311 in all code
- **PDM** - Package manager (not pip/poetry)
- **Dependencies** - Always check existing imports before adding new ones

### Code Quality Tools
- **Ruff** - Linting (Google docstring convention)
- **Black** - Code formatting  
- **MyPy** - Type checking (strict mode)
- **Bandit** - Security scanning
- **Coverage** - 90%+ unit test coverage required

### Style Guidelines
- **Line Length**: 100 characters
- **Docstrings**: Google style, imperative mood for functions
- **Type Hints**: Required for all function signatures
- **No Comments**: Code should be self-documenting unless explicitly requested
- **Imports**: Absolute imports, grouped per PEP8

### File Organization
```
ols/
├── app/           # FastAPI application (main.py, routers, endpoints)
├── src/           # Core business logic
│   ├── auth/      # Authentication (k8s, noop)
│   ├── cache/     # Conversation caching (memory, postgres)
│   ├── llms/      # LLM providers and loading
│   ├── query_helpers/ # Query processing utilities
│   └── utils/     # Shared utilities
├── constants.py   # Global constants
└── utils/         # Configuration and utilities
```

## Testing Strategy

### Test Structure
```
tests/
├── unit/          # Unit tests (pytest, mocking)
├── integration/   # Integration tests with real services  
├── e2e/           # End-to-end tests against running service
└── benchmarks/    # Performance benchmarks
```

### Testing Practices
- **pytest** - Testing framework
- **pytest-asyncio** - For async test support
- **Mock extensively** - Use unittest.mock for external dependencies
- **Fixtures** - Use conftest.py for shared test setup
- **Coverage** - Aim for 90%+ coverage, measured per test type
- **Test Naming** - `test_<function>_<scenario>` pattern

### Test Commands
```bash
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-e2e          # End-to-end tests  
make test              # All tests
make coverage-report   # Generate HTML coverage report
```

## Configuration Patterns

### Config Structure
- **YAML-based** - Primary config in `olsconfig.yaml`
- **Pydantic Models** - All config classes in `ols/app/models/config.py`
- **Environment Variables** - `OLS_CONFIG_FILE` for config path
- **Validation** - Extensive validation with custom validators

### Provider Configuration
LLM providers follow consistent patterns:
```yaml
llm_providers:
  - name: provider_name
    type: openai|azure_openai|watsonx|bam|...
    url: "api_endpoint"  
    credentials_path: path/to/api/key
    models:
      - name: model_name
```

## Development Workflow

### Setup
```bash
make install-deps     # Install all dependencies
make run              # Start development server
```

### Code Quality
```bash
make format           # Format code (black + ruff)
make verify           # Run all linters and type checks
make check-types      # MyPy type checking only
make security-check   # Bandit security scan
```

### Key Development Practices
1. **Always run linters** - `make verify` before commits
2. **Type everything** - Use proper type hints throughout
3. **Test thoroughly** - Write tests for new functionality
4. **Follow patterns** - Study existing code for consistency
5. **Configuration validation** - Use Pydantic models for all config
6. **Error handling** - Use custom exception classes
7. **Async/await** - Use async patterns for I/O operations
8. **Security first** - Never log secrets, use secure patterns

### Code Review Checklist
- [ ] Type hints on all functions
- [ ] Tests cover new functionality
- [ ] Docstrings follow Google convention
- [ ] No hardcoded credentials or secrets
- [ ] Follows existing naming patterns
- [ ] Proper error handling with custom exceptions
- [ ] Async/await used appropriately
- [ ] Configuration uses Pydantic models

## Common Patterns

### Error Handling
```python
from ols.utils.errors import SomeCustomError

def validate_something(data: str) -> None:
    if not data:
        raise SomeCustomError("Validation failed")
```

### Configuration Classes
```python
from pydantic import BaseModel, field_validator

class ConfigClass(BaseModel):
    field: str
    
    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        return value
```

### Provider Interface
Study existing providers in `ols/src/llms/providers/` for implementation patterns.

### Testing with Mocks
```python
from unittest.mock import Mock, patch
import pytest

@patch("ols.module.external_dependency")
def test_function(mock_dep):
    mock_dep.return_value = "expected_result"
    # test logic
```

## Performance Considerations
- Use async/await for I/O operations
- Cache expensive operations appropriately  
- Monitor token usage and context windows
- Consider memory usage for conversation history

## Security Guidelines
- Never commit secrets or API keys
- Use environment variables and file-based secrets
- Validate all inputs with Pydantic
- Use TLS for all external communications
- Follow principle of least privilege

Run `make help` to see all available commands.