# OpenShift LightSpeed Service - Development Guide for AI

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
- **uv** - Package manager (not pip/poetry/pdm)
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
- **Imports**: Absolute imports, grouped per PEP8, always at module level — inline imports only when required (e.g. avoiding circular imports or deferring an optional heavy dependency)

### File Organization
```
ols/
├── app/               # FastAPI application (main.py, routers, endpoints)
├── src/               # Core business logic
│   ├── auth/          # Authentication (k8s, noop)
│   ├── cache/         # Conversation caching (memory, postgres)
│   ├── llms/          # LLM providers and loading
│   ├── query_helpers/ # Query processing utilities
│   ├── mcp/           # Model Context Protocol support
│   ├── quota/         # Quota management
│   ├── rag_index/     # RAG indexing
│   └── utils/         # Shared utilities
├── constants.py       # Global constants
└── utils/             # Configuration and utilities
```

## Testing

```
tests/
├── unit/          # Unit tests (pytest, mocking)
├── integration/   # Integration tests with real services
├── e2e/           # End-to-end tests against running service
└── benchmarks/    # Performance benchmarks
```

- **pytest-asyncio** - Required for async tests; don't forget the `@pytest.mark.asyncio` decorator
- **Test naming** - `test_<function>_<scenario>` pattern
- **Coverage target** - 90%+ per test type

```bash
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-e2e           # End-to-end tests
make test               # All tests
make coverage-report    # Generate HTML coverage report
```

## Configuration

- **Config file** - `olsconfig.yaml`, user-specific, contains secrets, not in repo
- **Config location** - Use `OLS_CONFIG_FILE` env var to specify path, or ask the user
- **Examples** - See `examples/olsconfig.yaml` and `scripts/olsconfig.yaml`
- **Pydantic models** - All config classes in `ols/app/models/config.py`

## Development Workflow

```bash
make install-deps   # Install all dependencies
make run            # Start development server
make format         # Format code (black + ruff)
make verify         # Run all linters and type checks
make check-types    # MyPy type checking only
make security-check # Bandit security scan
```

## Detailed References

You MUST read the relevant file before working in a specific area — don't skip these:

- Adding or modifying an LLM provider → `docs/ai/providers.md`
- Adding or modifying config models → `docs/ai/config.md`
- Writing or debugging tests → `docs/ai/testing.md`
- Creating commits or pull requests → `docs/ai/git.md`

## Common Patterns

### Error Handling
Custom exceptions are defined in their respective domain modules:

```python
from ols.src.cache.cache_error import CacheError
from ols.src.quota.quota_exceed_error import QuotaExceedError
from ols.src.llms.llm_loader import LLMConfigurationError
from ols.utils.token_handler import PromptTooLongError
from ols.utils.checks import InvalidConfigurationError
```

### KISS and YAGNI
Implement the simplest solution for current requirements. No abstractions or flexibility for hypothetical future needs.

### Readability over cleverness
Write for a human reviewer verifying correctness. Avoid dense, clever constructs.

### Interface contracts first
Define function signatures and data shapes before implementing. Prefer explicit contracts over implicit ones.

### Test quality over test quantity
Assert specific values and behaviors, not just that code runs without error.

### Respect architectural boundaries
Do not cross module or layer boundaries, even when it is the shorter path.

### Keep changes scoped
Do not refactor, rename, or reformat outside the direct path of the task. Unrelated improvements belong in a separate PR.

## Maintaining This Guide

Update this file and the relevant `docs/ai/` reference when patterns, tooling, or conventions change, or after repeated mistakes. Only document what you would get wrong without being told — remove anything inferable from reading the code.
