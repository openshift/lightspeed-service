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

### Pre-commit hook settings

It is possible to run formatters and linters automatically for all commits. You just need
to copy file `hooks/pre-commit` into subdirectory `.git/hooks/`. It must be done manually
because the copied file is an executable script (so from GIT point of view it is unsafe
to enable it automatically).


### Code coverage measurement

During testing, code coverage is measured. If the coverage is below defined threshold (see `pyproject.toml` settings for actual value), tests will fail. We measured and checked code coverage in order to be able to develop software with high quality.

Code coverage reports are generated in JSON and also in format compatible with _JUnit_. It is also possible to start `make coverage-report` to generate code coverage reports in form of interactive HTML pages. These pages are stored in `htmlcov` subdirectory. Just open index page from this subdirectory in your web browser.



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
