# Updating Konflux Hermetic Requirements

This document describes how to regenerate the hashed requirements files used by the Konflux hermetic build pipeline.

## Prerequisites

Standard `uv` package manager (installed via `pip install uv` or available in the project venv).

## Running

```bash
make konflux-requirements
```

This runs `python3 scripts/konflux_resolve.py --profile cpu`, which:

1. Resolves all dependencies from `pyproject.toml` using `uv pip compile` with manual overrides.
2. Loads the RHOAI index and auto-generates version overrides for all RHOAI-available packages.
3. Re-resolves with both manual and auto-generated overrides to pin RHOAI versions.
4. Classifies each package by checking the RHOAI index: RHOAI wheel, PyPI sdist, or PyPI wheel (last resort).
5. Fetches SHA-256 hashes for every resolved package.
6. Writes hashed requirements files to `.konflux/`.
7. Generates build dependencies via `pybuild-deps`.
8. Patches `.tekton/` pipeline YAML files with the updated binary packages list.

## Output files

| File | Description |
|------|-------------|
| `.konflux/requirements.hashes.wheel.txt` | RHOAI wheel packages with hashes |
| `.konflux/requirements.hashes.source.txt` | PyPI source (sdist) packages with hashes |
| `.konflux/requirements.hashes.wheel.pypi.txt` | PyPI wheel packages with hashes (no sdist available) |
| `.konflux/requirements-build.txt` | Build-time dependencies for source packages |

These files are referenced by `.tekton/lightspeed-service-pull-request.yaml` and `.tekton/lightspeed-service-push.yaml`.

## Configuration

**`.konflux/profiles.toml`** defines the build profile (RHOAI index URL, target platforms, tekton files, bootstrap packages).

**`.konflux/requirements.overrides.txt`** pins specific package versions to match what is available on the RHOAI index. These manual overrides take precedence over auto-generated ones and are used when a package needs a specific version (e.g., to avoid dependency conflicts).

**`.konflux/pypi_wheel_only.txt`** lists packages that only have wheel distributions on PyPI (no sdist). The script auto-detects these and warns; adding them here suppresses the warning.

## Verbose output

For debugging, run directly with `--verbose`:

```bash
python3 scripts/konflux_resolve.py --profile cpu --verbose
```
