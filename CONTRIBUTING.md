# CONTRIBUTING

## TLDR;

1. Create your own fork of the repo
2. Make changes to the code in your fork
3. Run unit tests and integration tests
4. Check the code with linters
5. Submit PR from your fork to main branch of the project repo

## Setting up your development environment

The development prefers [Python 3.11](https://docs.python.org/3/whatsnew/3.11.html) or later due to significant improvement on performance, optimizations which benefit modern ML, AI, LLM, NL stacks, and improved asynchronous proccessing capabilities.

```bash
# clone your fork
git clone https://github.com/YOUR-GIT-PROFILE/lightspeed-service.git

# move into the directory
cd lightspeed-service

# setup your python virtual environment to avoid conflicts with your
# system packages
python3.11 -m venv venv

# activate the virtual environment
source ./venv/bin/activate

# upgrade pip to the most recent version
pip install --upgrade pip

# install project dependencies
make install-deps

# install dev/tests dependencies
make install-deps-test

# run all tests
make test

# code formatting
# (this is also run automatically as part of pre-commit hook)
make format

# code style and docstring style
# (this is also run automatically as part of pre-commit hook)
make verify

```

Happy hacking!


## Definition of Done

### A deliverable is to be considered “done” when

* Code is complete, commented, and merged to the relevant release branch
* User facing documentation written (where relevant)
* Acceptance criteria in the related Jira ticket(where applicable) are verified and fulfilled
* Pull request title+commit includes JIRA number
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
test-unit                 Run the unit tests
test-integration          Run integration tests tests
test-e2e                  Run end to end tests
```

All tests are based on [Pytest framework](https://docs.pytest.org/en/) and code coverage is measured by plugin [pytest-cov](https://github.com/pytest-dev/pytest-cov). For mocking and patching, the [unittest framework](https://docs.python.org/3/library/unittest.html) is used.

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
        LLMLoader(provider=None)
```

### Checking what was printed and logged to stdout or stderr by the tested code

It is possible to capture stdour and stderr by using standard fixture `capsys`:

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

If updating `requirements.txt` follow the guidance for "main" branch. If a dependency is no longer required, remove it from the list.

***Note:*** *If cutting a release branch freeze the `requirements.txt` list as described in the corresponding section below.*

### For "main" branch
- The "main" branch is the development branch and we expect to be moving forward and taking advantages of the latest releases of libraries, etc. When updating `requirements.txt` on main branch ONLY include the main dependency without version or with minimum version >= X, but DO NOT specify fixed versions or sub-depdencies

```bash
# Good definitions on main branch
langchain
langchain>=0.0.335

# Bad definitions on main branch (fixed version)
langchain==0.0.335
```

### For a release branch
- A release branch or tag is expected to always lead to identical results. For this, when cutting a release branch create a prescriptive versioned `requirements.txt`

```bash
# Create dependency list with fixed versions
pip freeze -l --isolated > requirements.txt
```

This will create a requirements file with the exact versions of the main dependencies and their corresponding sub-dependencies. 

```bash
# Example for a requirements.txt for a release
# using fixed versions for all dependencies
ibm-generative-ai==0.5.0
idna==3.4
jsonpatch==1.33
jsonpointer==2.4
langchain==0.0.335
langsmith==0.0.64
marshmallow==3.20.1
multidict==6.0.4
mypy-extensions==1.0.0
numpy==1.26.2
# ...more entries
```

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
    """
```

For further guidance, see the rest of our codebase, or check sources online. There are many, eg. [this one](https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e).
