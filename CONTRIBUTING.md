# CONTRIBUTING

## TLDR;

1. Create your own fork of the repo
2. Make changes to the code in your fork
3. Submit PR from your fork to main branch of the project repo

## System prerequisities

These tools needs the be present in your system

- `poetry` - dependency management, [install guide](https://python-poetry.org/docs/)
- `podman` - image builder, [install guide](https://podman.io/docs/installation)

## Setting up your development environment

The development prefers [Python 3.11](https://docs.python.org/3/whatsnew/3.11.html) or later due to significant improvement on performance, optimizations which benefit modern ML, AI, LLM, NL stacks, and improved asynchronous proccessing capabilities.

```bash
# clone your fork
git clone https://github.com/williamcaban/lightspeed-service.git

# move into the directory
cd lightspeed-service

# ensure poetry uses correct python version for venv (in case its not a default in your system)
poetry env use `which python3.11`

# spawn a shell/virtual env
poetry shell

# install project and its dependencies (from lock)
poetry install --with=dev

# code formatting (Run this as a pre-commit step for your code changes)
make format
```

Happy hacking!

## Updating Dependencies

Do `poetry add dep` for adding main dependency or `poetry add --group=dev dep` for adding development dependency.

Do `poetry lock` after adding/bumping dependency to regenerate lock file.

## CI/CD

We use OpenShift PROW. Its configration is stored [here](https://github.com/openshift/release/blob/master/ci-operator/config/openshift/lightspeed-service/openshift-lightspeed-service-main.yaml).
