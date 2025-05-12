"""Plugins."""

import importlib.util
import pathlib
import sys


def _import_modules_from_dir(dir_name: str) -> None:
    """Import all modules from the provided directory.

    The path is either relative to the file where this function is
    called or absolute.
    """
    plugins_dir = pathlib.Path(__file__).parent.resolve() / dir_name
    sys.path.append(plugins_dir.as_posix())

    # gather .py files except files starting with "__"
    modules = [
        f
        for f in plugins_dir.iterdir()
        if f.suffix == ".py" and not f.stem.startswith("__")
    ]

    for module_name in modules:
        spec = importlib.util.spec_from_file_location(
            "module_name", module_name.as_posix()
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            raise Exception("Can not import module %s", module_name)

# TODO: move providers to plugins
# TODO: move tools to plugins
