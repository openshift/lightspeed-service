# Python dependency bumps

Read this before adding or bumping a package. Humans: [CONTRIBUTING.md](../../CONTRIBUTING.md#updating-dependencies). Hermetic regen: [docs/update-requirements.md](../update-requirements.md).

Konflux does **not** install from `uv.lock`. A PR that only changes the lock (or `pyproject.toml`) will fail prefetch or ship the old hermetic pins.

## Sequence

1. One package (including a CVE bump): `make bump PKG=PACKAGE VERSION=VERSION`. It regenerates `.konflux/`, `uv.lock`, and `requirements.txt` together, preferring the currently locked versions of every other package. Not `make update-deps` / `uv lock --upgrade` (those upgrade everything and skip the hermetic regen).
2. Transitive CVE: same command — do **not** add `PACKAGE==VERSION` to `pyproject.toml`. Exact `==` can downgrade a newer RHOAI wheel (e.g. lock needed `0.8.0` while hermetic already had `0.8.15`). If a floor is required, `>=VERSION` like the existing CVE comments.
3. Direct dep: `make bump` raises the `>=` floor in `pyproject.toml` itself.
4. If the fixed version is missing from the RHOAI index, add the pin to `.konflux/requirements.overrides.txt` and `[tool.uv] override-dependencies` in `pyproject.toml`, and list wheel-only packages in `.konflux/pypi_wheel_only.txt`, **before** running `make bump`. Those are deliberate policy inputs `make bump` never modifies.
5. Confirm `.konflux/requirements-build.txt` has **no** `--index-url`. Wheel hashes **do** use the RHOAI `packages.redhat.com` index; build deps must stay on public PyPI. A leaked `console.redhat.com` index makes Hermeto reject packages that exist on PyPI (`pyproject-metadata==…`).

`make bump` scrubs the uv index configuration itself; no manual `unset`/`UV_NO_CONFIG=1` preamble is needed.

Do not skip regen when the script warns about pin conflicts. A dirty `requirements-build.txt` is a red prefetch, not a partial success.

Do not edit `uv.lock` by hand.
