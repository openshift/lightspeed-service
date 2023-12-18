# CONTRIBUTING

## TLDR;

1. Create your own fork of the repo
2. Make changes to the code in your fork
3. Submit PR from your fork to main branch of the project repo

## Setting up your development environment

The development prefers [Python 3.11](https://docs.python.org/3/whatsnew/3.11.html) or later due to significant improvement on performance, optimizations which benefit modern ML, AI, LLM, NL stacks, and improved asynchronous proccessing capabilities.

```bash
# Clone your fork
git clone https://github.com/williamcaban/lightspeed-service.git

# move into the directory
cd lightspeed-service

# Setup your Python Virtual Environment 
# to avoid conflicts with your system packages
python3.11 -m venv venv

# Activate the virtual environment
source ./venv/bin/activate

# Upgrade pip to the most recent version
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Code formatting (run this as a pre-commit step for your code changes)
make format

# Code style and docstring style (run this as a pre-commit step for your code changes)
make verify

```

Happy hacking!

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
