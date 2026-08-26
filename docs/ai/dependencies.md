# Python dependency bumps

Read this before adding or bumping a package. Humans: [CONTRIBUTING.md](../../CONTRIBUTING.md#updating-dependencies). Hermetic regen: [docs/update-requirements.md](../update-requirements.md).

Konflux does **not** install from `uv.lock`. A PR that only changes the lock (or `pyproject.toml`) will fail prefetch or ship the old hermetic pins.

## Sequence

1. One package: `uv lock --upgrade-package PACKAGE`. Not `make update-deps` / `uv lock --upgrade` (those upgrade everything).
2. Transitive CVE: do **not** add `PACKAGE==VERSION` to `pyproject.toml`. Prefer the lock-only bump. If a floor is required, `>=VERSION` like the existing CVE comments. Exact `==` can downgrade a newer RHOAI wheel (e.g. lock needed `0.8.0` while hermetic already had `0.8.15`).
3. Direct dep: raise the `>=` floor, then `--upgrade-package`.
4. `make konflux-requirements` with a clean uv config (unset `UV_CONFIG_FILE` and index env vars; `UV_NO_CONFIG=1`). Do not export `UV_CONFIG_FILE` empty.
5. Confirm `.konflux/requirements-build.txt` has **no** `--index-url`. Wheel hashes **do** use the RHOAI `packages.redhat.com` index; build deps must stay on public PyPI. A leaked `console.redhat.com` index makes Hermeto reject packages that exist on PyPI (`pyproject-metadata==…`).

Do not skip regen when the script warns about pin conflicts. A dirty `requirements-build.txt` is a red prefetch, not a partial success.

Do not edit `uv.lock` by hand.
