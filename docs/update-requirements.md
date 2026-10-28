# Updating Konflux Hermetic Requirements

This document describes how to regenerate the hashed requirements files used by the Konflux hermetic build pipeline.

For the full bump sequence (single-package and full refreshes), see [CONTRIBUTING.md](../CONTRIBUTING.md#updating-dependencies). `make bump` regenerates the files described here and refreshes `uv.lock` and `requirements.txt` in the same run.

## Prerequisites

- A recent uv. The make target provisions Python 3.12 and the resolver's dependencies (packaging, tomlkit, pybuild-deps, pip-tools, pip) itself via `uv run --no-project`.
- Podman or Docker installed and usable. The resolver prefers Podman when both are available.
- Access to pull `quay.io/syedriko/uv:prefer-index`. Dependency resolution runs in this container, whose uv supports `--index-strategy prefer-index`; it does not use the host's standard uv for this step.
- Network access to the configured RHOAI index and public PyPI.

Regen must **not** inherit a RHOAI/default extra index. The sandbox image and some laptops ship a `uv.toml` whose default index is RHOAI (`console.redhat.com` or similar). The host-side `pybuild-deps` step can then write that `--index-url` into `.konflux/requirements-build.txt`, and Hermeto prefetch fails (`PackageRejected: No distributions found for package …`).

`make bump` guards against this: it scrubs `UV_CONFIG_FILE` and the index variables and runs every step with `UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null`. Only if you invoke `scripts/konflux_resolve.py` directly do you need the clean environment yourself:

```bash
unset UV_CONFIG_FILE UV_INDEX_URL UV_DEFAULT_INDEX PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export UV_NO_CONFIG=1
```

Do not `export UV_CONFIG_FILE=""`. Unset it. `UV_NO_CONFIG=1` (or `uv --no-config`) ignores file config as well.

## Running

```bash
make bump                                  # refresh all dependencies
make bump PKG=<package> VERSION=<minimum>  # single package (e.g. a CVE bump)
```

(`make konflux-requirements` remains an alias for `make bump`.)

This runs `scripts/konflux_resolve.py --profile cpu` under `uv run --no-project` with the resolver's tools, then `uv lock` (or `uv lock -P '<package>>=<version>'` for a targeted bump), then `uv export` to `requirements.txt` — all with the scrubbed config. The resolver:

1. Resolves production dependencies from `pyproject.toml` in the uv container using `uv pip compile --index-strategy prefer-index`, the configured RHOAI index, and public PyPI as the default index. Applies manual overrides from `.konflux/requirements.overrides.txt` and any extras configured in the profile; development dependency groups are not included.
2. Uses uv's emitted index annotations to identify whether each resolved package came from RHOAI or PyPI. There is no auto-generated override file or second resolution pass.
3. Fetches SHA-256 hashes from each package's selected index. Pins and hashes the bootstrap tools from RHOAI, requiring wheels for every configured target architecture.
4. Classifies packages into RHOAI wheels, PyPI source distributions (sdists), or PyPI wheels as a last resort when no sdist is available or the package is listed as wheel-only.
5. Writes the hashed runtime and bootstrap requirements files to `.konflux/`.
6. Discovers source-build requirements for PyPI source packages (host-side `pybuild-deps`) and resolves them together with the runtime set. Build dependencies resolved from RHOAI are written to `requirements.hermetic.txt`; the rest are written hashed to the index-free `requirements-build.txt`.
7. Patches `.tekton/` pipeline YAML files with the updated binary packages list.

## Output files

| File | Description |
|------|-------------|
| `.konflux/requirements.hashes.wheel.txt` | RHOAI wheel packages with hashes |
| `.konflux/requirements.hashes.source.txt` | PyPI source (sdist) packages with hashes |
| `.konflux/requirements.hashes.wheel.pypi.txt` | PyPI wheel packages with hashes (no sdist available or explicitly listed as wheel-only) |
| `.konflux/requirements.hermetic.txt` | Pinned, hashed RHOAI wheels — bootstrap tools (`maturin`, `uv`, `uv-build`) plus source-build dependencies resolved from RHOAI — with wheels for all configured target architectures |
| `.konflux/requirements-build.txt` | Build-time dependencies for source packages. **PyPI only** — must not contain `--index-url`. |

These files are referenced by `.tekton/lightspeed-service-pull-request.yaml` and `.tekton/lightspeed-service-push.yaml` for dependency prefetch. `requirements.hermetic.txt` makes the bootstrap tools available offline during hermetic builds.

After regen, check `requirements-build.txt`. `.konflux/requirements.hashes.wheel.txt` **should** start with `--index-url https://packages.redhat.com/api/pypi/public-rhai/...` (RHOAI wheels). `requirements-build.txt` must **not**. If it gained `--index-url` (especially `console.redhat.com`), discard those generated files and rerun with a clean uv config. `make verify` / `scripts/verify_hermetic_requirements.sh` rejects a leaked index on that file.

Also check that packages in `requirements.hashes.source.txt` are **not** in the `.tekton/` pipelines' `binary.packages` lists. Hermeto uses that list across all requirements files: a source requirement listed as binary will fetch a PyPI wheel and fail Enterprise Contract's disallowed-binary-attribute rule. The verification script checks both pipelines for this overlap. The source requirements file may contain both sdist and wheel hashes; its filename or hashes alone do not force Hermeto to choose the sdist.

After regeneration, verify that all runtime dependency names are accounted for and review the generated changes:

```bash
bash scripts/verify_hermetic_requirements.sh
git diff -- .konflux/ .tekton/
```

Konflux installs from these generated requirements, not `uv.lock`. The verification script checks package-name coverage against the runtime export of `uv.lock`, with a small allowlist for legitimate differences; it does not require identical version pins between the two resolutions.

## Configuration

**`.konflux/profiles.toml`** defines the build profile (RHOAI index URL, target Python version and platforms, Tekton files, bootstrap packages, optional extras, and output suffix). The CPU profile currently uses the RHOAI 3.5 CPU UBI9 index and writes unsuffixed output files.

**`.konflux/requirements.overrides.txt`** supplies manual version pins or constraints to uv during resolution. Use it for compatibility constraints or security fixes, including versions that require a PyPI fallback when the RHOAI index does not provide a suitable version. The resolver does not generate automatic overrides.

**`.konflux/pypi_wheel_only.txt`** lists packages that must be fetched as PyPI wheels instead of built from source: no sdist on PyPI, C extensions that cannot be compiled in the hermetic environment, or sdist build requirements that cannot be jointly satisfied with the other source builds. The script auto-detects missing sdists and warns; listing a package here classifies it explicitly.

## Verbose output

For debugging, run directly with `--verbose`:

```bash
env -u UV_CONFIG_FILE -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u UV_INDEX \
	-u UV_EXTRA_INDEX_URL -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
	UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null \
	uv run --no-project --python 3.12 --with packaging --with tomlkit \
	--with pybuild-deps --with pip-tools --with pip \
	python scripts/konflux_resolve.py --profile cpu --verbose
```

Direct invocation needs the clean environment shown above — the scrubbing lives in the `make bump` recipe, not in the script.
