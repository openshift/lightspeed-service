# Updating Konflux Hermetic Requirements

This document describes how to regenerate the hashed requirements files used by the Konflux hermetic build pipeline.

For the full bump sequence (when to change `pyproject.toml`, `uv lock --upgrade-package`, and this regen), see [CONTRIBUTING.md](../CONTRIBUTING.md#updating-dependencies).

## Prerequisites

Standard `uv` (`pip install uv` or the project venv). Regen must **not** inherit a RHOAI/default extra index. The sandbox image and some laptops ship a `uv.toml` whose default index is RHOAI (`console.redhat.com` or similar). `pybuild-deps` will then write that `--index-url` into `.konflux/requirements-build.txt`, and Hermeto prefetch fails (`PackageRejected: No distributions found for package …`).

Before `make konflux-requirements`:

```bash
unset UV_CONFIG_FILE UV_INDEX_URL UV_DEFAULT_INDEX PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export UV_NO_CONFIG=1
```

Do not `export UV_CONFIG_FILE=""`. Unset it. `UV_NO_CONFIG=1` (or `uv --no-config`) ignores file config as well.

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
| `.konflux/requirements-build.txt` | Build-time dependencies for source packages. **PyPI only** — must not contain `--index-url`. |

These files are referenced by `.tekton/lightspeed-service-pull-request.yaml` and `.tekton/lightspeed-service-push.yaml`.

After regen, check `requirements-build.txt`. `.konflux/requirements.hashes.wheel.txt` **should** start with `--index-url https://packages.redhat.com/api/pypi/public-rhai/...` (RHOAI wheels). `requirements-build.txt` must **not**. If it gained `--index-url` (especially `console.redhat.com`), discard those generated files and rerun with a clean uv config. `make verify` / `scripts/verify_hermetic_requirements.sh` rejects a leaked index on that file.

## Configuration

**`.konflux/profiles.toml`** defines the build profile (RHOAI index URL, target platforms, tekton files, bootstrap packages).

**`.konflux/requirements.overrides.txt`** pins specific package versions to match what is available on the RHOAI index. These manual overrides take precedence over auto-generated ones and are used when a package needs a specific version (e.g., to avoid dependency conflicts).

**`.konflux/pypi_wheel_only.txt`** lists packages that only have wheel distributions on PyPI (no sdist). The script auto-detects these and warns; adding them here suppresses the warning.

## Verbose output

For debugging, run directly with `--verbose`:

```bash
python3 scripts/konflux_resolve.py --profile cpu --verbose
```
