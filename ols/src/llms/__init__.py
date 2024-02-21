"""Interface to LLMs."""

# NOTE: In order to register all providers automatically, we are importing
# everything from the providers package.

import importlib.util
import pathlib
import sys


def import_providers():
    """Import all providers from the providers directory."""
    providers_dir = pathlib.Path(__file__).parent.resolve() / "providers"
    sys.path.append(providers_dir.as_posix())
    modules = [f for f in providers_dir.iterdir() if not f.stem.startswith("__")]

    for module in modules:
        spec = importlib.util.spec_from_file_location("module_name", module.as_posix())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


import_providers()
