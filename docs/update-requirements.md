# Updating Konflux Hermetic Requirements

This document describes how to regenerate the hashed requirements files used by the Konflux hermetic build pipeline.

For the full bump sequence (when to change `pyproject.toml`, `uv lock --upgrade-package`, and this regen), see [CONTRIBUTING.md](../CONTRIBUTING.md#updating-dependencies).

## Prerequisites

Run from the repository root with:

- Python 3.12 and the resolver's Python dependencies installed in the environment used by `python3`.
- Podman or Docker installed and usable. The resolver prefers Podman when both are available.
- Access to pull `quay.io/syedriko/uv:prefer-index`. Dependency resolution runs in this container, whose uv supports `--index-strategy prefer-index`; it does not use the host's standard uv for this step.
- `pybuild-deps` installed on the host and available on `PATH` to generate build-time dependencies for PyPI source packages.
- Network access to the configured RHOAI index and public PyPI.

Regen must **not** inherit a RHOAI/default extra index. The sandbox image and some laptops ship a `uv.toml` whose default index is RHOAI (`console.redhat.com` or similar). The host-side `pybuild-deps` step can then write that `--index-url` into `.konflux/requirements-build.txt`, and Hermeto prefetch fails (`PackageRejected: No distributions found for package …`).

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

1. Resolves production dependencies from `pyproject.toml` in the uv container using `uv pip compile --index-strategy prefer-index`, the configured RHOAI index, and public PyPI as the default index. Applies manual overrides from `.konflux/requirements.overrides.txt` and any extras configured in the profile; development dependency groups are not included.
2. Uses uv's emitted index annotations to identify whether each resolved package came from RHOAI or PyPI. There is no auto-generated override file or second resolution pass.
3. Fetches SHA-256 hashes from each package's selected index. Pins and hashes the bootstrap tools from RHOAI, requiring wheels for every configured target architecture.
4. Classifies packages into RHOAI wheels, PyPI source distributions (sdists), or PyPI wheels as a last resort when no sdist is available or the package is listed as wheel-only.
5. Writes the hashed runtime and bootstrap requirements files to `.konflux/`.
6. Runs host-side `pybuild-deps` to generate build dependencies for PyPI source packages, then removes entries already supplied by RHOAI wheels or bootstrap tools.
7. Patches `.tekton/` pipeline YAML files with the updated binary packages list.

## Output files

| File | Description |
|------|-------------|
| `.konflux/requirements.hashes.wheel.txt` | RHOAI wheel packages with hashes |
| `.konflux/requirements.hashes.source.txt` | PyPI source (sdist) packages with hashes |
| `.konflux/requirements.hashes.wheel.pypi.txt` | PyPI wheel packages with hashes (no sdist available or explicitly listed as wheel-only) |
| `.konflux/requirements.hermetic.txt` | Pinned, hashed RHOAI bootstrap tools (`maturin`, `uv`, `uv-build`) with wheels for all configured target architectures |
| `.konflux/requirements-build.txt` | Build-time dependencies for source packages. **PyPI only** — must not contain `--index-url`. |

These files are referenced by `.tekton/lightspeed-service-pull-request.yaml` and `.tekton/lightspeed-service-push.yaml` for dependency prefetch. `requirements.hermetic.txt` makes the bootstrap tools available offline during hermetic builds.

After regen, check `requirements-build.txt`. `.konflux/requirements.hashes.wheel.txt` **should** start with `--index-url https://packages.redhat.com/api/pypi/public-rhai/...` (RHOAI wheels). `requirements-build.txt` must **not**. If it gained `--index-url` (especially `console.redhat.com`), discard those generated files and rerun with a clean uv config. `make verify` / `scripts/verify_hermetic_requirements.sh` rejects a leaked index on that file.

After regeneration, verify that all runtime dependency names are accounted for and review the generated changes:

```bash
bash scripts/verify_hermetic_requirements.sh
git diff -- .konflux/ .tekton/
```

Konflux installs from these generated requirements, not `uv.lock`. The verification script checks package-name coverage against the runtime export of `uv.lock`, with a small allowlist for legitimate differences; it does not require identical version pins between the two resolutions.

## Configuration

**`.konflux/profiles.toml`** defines the build profile (RHOAI index URL, target Python version and platforms, Tekton files, bootstrap packages, optional extras, and output suffix). The CPU profile currently uses the RHOAI 3.5 CPU UBI9 index and writes unsuffixed output files.

**`.konflux/requirements.overrides.txt`** supplies manual version pins or constraints to uv during resolution. Use it for compatibility constraints or security fixes, including versions that require a PyPI fallback when the RHOAI index does not provide a suitable version. The resolver does not generate automatic overrides.

**`.konflux/pypi_wheel_only.txt`** lists packages that only have wheel distributions on PyPI (no sdist). The script auto-detects these and warns; adding them here suppresses the warning.

## Verbose output

For debugging, run directly with `--verbose`:

```bash
python3 scripts/konflux_resolve.py --profile cpu --verbose
```
