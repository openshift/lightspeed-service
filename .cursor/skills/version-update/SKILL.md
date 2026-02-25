---
name: version-update
description: Updates the OLS version number across all required files. Use when bumping the version for a release or when the user asks to update, bump, or change the version number.
disable-model-invocation: true
---

# Version Update

Update the version in **three** files. They must all match.

## Files to update

**1. `ols/version.py`** — source of truth

```python
__version__ = "X.Y.Z"
```

**2. `docs/openapi.json`** — OpenAPI spec

```json
"info": {
    "version": "X.Y.Z"
}
```

**3. `scripts/build-container.sh`** — container build (note the `v` prefix)

```bash
OLS_VERSION=vX.Y.Z
```

## Checklist

- [ ] `ols/version.py` updated
- [ ] `docs/openapi.json` updated
- [ ] `scripts/build-container.sh` updated with `v` prefix
- [ ] All three versions match
- [ ] Regenerate OpenAPI docs if needed after version change
