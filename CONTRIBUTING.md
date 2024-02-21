# CONTRIBUTING

## TLDR;

1. Create your own fork of the repo
2. Make changes to the code in your fork
3. Run unit tests and integration tests
4. Check the code with linters
5. Submit PR from your fork to main branch of the project repo

## Setting up your development environment

The development requires [Python 3.11](https://docs.python.org/3/whatsnew/3.11.html) due to significant improvement on performance, optimizations which benefit modern ML, AI, LLM, NL stacks, and improved asynchronous processing capabilities.

```bash
# clone your fork
git clone https://github.com/YOUR-GIT-PROFILE/lightspeed-service.git

# move into the directory
cd lightspeed-service

# setup your environment with pdm
pdm install

# now either do (with backticks)
`pdm venv activate`
# which will active the venv where you can further work, or prefix the rest of commands with `pdm run`, eg. `pdm run make test`

# run unit+integration tests (e2e test requires a running OLS instance)
make test-unit && make test-integration

# code formatting
# (this is also run automatically as part of pre-commit hook if configured)
make format

# code style and docstring style
# (this is also run automatically as part of pre-commit hook if configured)
make verify

# check type hints
# (this is also run automatically as part of pre-commit hook)
make check-types
```

Happy hacking!


## Definition of Done

### A deliverable is to be considered “done” when

* Code is complete, commented, and merged to the relevant release branch
* User facing documentation written (where relevant)
* Acceptance criteria in the related Jira ticket (where applicable) are verified and fulfilled
* Pull request title+commit includes Jira number
* Changes are covered by unit tests that run cleanly in the CI environment (where relevant)
* Changes are covered by integration tests that run cleanly in the CI environment (where relevant)
* Changes are covered by E2E tests that run cleanly in the CI environment (where relevant)
* All linters are running cleanly in the CI environment
* Code changes reviewed by at least one peer
* Code changes acked by at least one project owner


### Pre-commit hook settings

It is possible to run formatters and linters automatically for all commits. You just need
to copy file `hooks/pre-commit` into subdirectory `.git/hooks/`. It must be done manually
because the copied file is an executable script (so from GIT point of view it is unsafe
to enable it automatically).


### Code coverage measurement

During testing, code coverage is measured. If the coverage is below defined threshold (see `pyproject.toml` settings for actual value), tests will fail. We measured and checked code coverage in order to be able to develop software with high quality.

Code coverage reports are generated in JSON and also in format compatible with _JUnit_. It is also possible to start `make coverage-report` to generate code coverage reports in form of interactive HTML pages. These pages are stored in `htmlcov` subdirectory. Just open index page from this subdirectory in your web browser.

Overall code coverage measured for both unit tests and integration tests can be checked by following command:

```
make check-coverage
```

The threshold set for combined coverage is larger than threshold for unit tests and integration tests because it is preferred to have most statements covered by at least one type of tests.



### Type hints checks

It is possible to check if type hints added into the code are correct and whether assignments, function calls etc. use values of the right type. This check is invoked by following command:

```
make check-types
```

Please note that type hints check might be very slow on the first run. Subsequent runs are much faster thanks to cache that Mypy uses.
This check is part of CI job that verifies sources.


### Linters

_Ruff_ tools is used as a linter. There are bunch of linters enabled for this repository. All of them are specified in `pyproject.toml` in section `[tool.ruff]`. Some specific rules can be disabled using `ignore` parameter (empty now). List of all linters recognized by Ruff can be retrieved by:

```
ruff linter
```

Description of all rules are available on https://docs.astral.sh/ruff/rules/


## Testing

Three group of software tests are used in this repository, each group from the test suite having different granularity. These groups are designed to represent three layers:

1. Unit Tests
1. Integration Tests
1. End to End tests (e2e)

Unit tests followed by integration tests can be started by using the following command:

```
make tests
```

It is also possible to run just one selected group of tests:

```
make test-unit                 Run the unit tests
make test-integration          Run integration tests tests
make test-e2e                  Run end to end tests
```

All tests are based on [Pytest framework](https://docs.pytest.org/en/) and code coverage is measured by plugin [pytest-cov](https://github.com/pytest-dev/pytest-cov). For mocking and patching, the [unittest framework](https://docs.python.org/3/library/unittest.html) is used.

Currently code coverage threshold for integration tests is set to 60%. This value is specified directly in Makefile, because the coverage threshold is different from threshold required for unit tests.

As specified in Definition of Done, new changes needs to be covered by tests.



## Tips and hints for developing unit tests

### Patching

For patching, for example introducing mock object instead of real object, it is
possible to use `patch` imported from `unittest.mock`. It is usually used as
decorator put before test implementation:

```python
@patch("redis.StrictRedis", new=MockRedis)
def test_conversation_cache_in_redis(redis_cache_config):
   ...
   ...
   ...
```

- `new=` allow us to use different function or class
- `return_value=` allow us to define return value (no mock will be called)

It is also possible to use it inside the test implementation as context manager:

```python
def test_xyz():
    ml = mock_llm_chain({"text": retval})
    with patch("ols.src.query_helpers.question_validator.LLMChain", new=ml):
        ...
        ...
        ...
```

### Verifying that some exception is thrown

Sometimes it is needed to test whether some exception is thrown from tested function or method. In this case `pytest.raises` can be used:


```python
def test_conversation_cache_wrong_cache(invalid_cache_type_config):
    """Check if wrong cache env.variable is detected properly."""
    with pytest.raises(ValueError):
        CacheFactory.conversation_cache(invalid_cache_type_config)
```

It is also possible to check if the exception is thrown with expected message. The message (or its part) is written as regexp:

```python
def test_constructor_no_provider():
    """Test that constructor checks for provider."""
    # we use bare Exception in the code, so need to check
    # message, at least
    with pytest.raises(Exception, match="ERROR: Missing provider"):
        load_llm(provider=None)
```

### Checking what was printed and logged to stdout or stderr by the tested code

It is possible to capture stdout and stderr by using standard fixture `capsys`:

```python
def test_foobar(capsys):
    """Test the foobar function that prints to stdout."""
    foobar("argument1", "argument2")

    # check captured log output
    captured_out = capsys.readouterr().out
    assert captured_out == "Output printed by foobar function"
    captured_err = capsys.readouterr().err
    assert captured_err == ""
```

Capturing logs:

```python
@patch.dict(os.environ, {"LOG_LEVEL": "INFO"})
def test_logger_show_message_flag(mock_load_dotenv, capsys):
    """Test logger set with show_message flag."""
    logger = Logger(logger_name="foo", log_level=logging.INFO, show_message=True)
    logger.logger.info("This is my debug message")

    # check captured log output
    # the log message should be captured
    captured_out = capsys.readouterr().out
    assert "This is my debug message" in captured_out

    # error output should be empty
    captured_err = capsys.readouterr().err
    assert captured_err == ""
```



## Updating Dependencies

We are using the [PDM tool](https://github.com/pdm-project/pdm) to manage our dependencies.

To add a new dependency do:
1. `pdm add mystery-package` - PDM will add it to `pyproject.toml` automatically
2. re-lock with `pdm lock` - this will update a `pdm.lock` with a new dependency

As we need to be Konflux-ready (https://redhat-appstudio.github.io/appstudio.docs.ui.io/), we need to have pinned versions in the `pyproject.toml`. If you added a new dependency without an explicit version pin, the PDM tool resolves its version in the lock. You need to search for the dependency you've added in the lock file and whatever version you find there, use it to pin it in `pyproject.toml`.


## Code style

### Docstrings style
We are using [Google's docstring style](https://google.github.io/styleguide/pyguide.html).

Here is simple example:
```python
def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
    """Example function with PEP 484 type annotations.
    
    Args:
        param1: The first parameter.
        param2: The second parameter.
    
    Returns:
        The return value. True for success, False otherwise.
    
    Raises:
        ValueError: If the first parameter does not contain proper model name
    """
```

For further guidance, see the rest of our codebase, or check sources online. There are many, eg. [this one](https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e).


## Adding a new provider/model
To add a new provider, follow these steps:

1. Create a new file in the `ols/src/llm/providers/` directory.
2. In this file, define a class that inherits from `LLMProvider`.
3. Register this class for use by decorating it with `@register_llm_provider_as("your_provider")`. You can refer to existing providers for examples.

Please note that your custom provider must adhere to the interface defined by `AbstractLLMProvider` to ensure proper integration. Specifically, you must define a `default_params` property and a `load` method in your custom provider class.

You'll also need to modify your `olsconfig.yaml` file to include an appropriate entry for your new provider.

Once you've created and registered your new provider as described above, no further code modifications are necessary. Our discovery mechanism will automatically locate your provider. After that, you'll be able to use it as needed.
